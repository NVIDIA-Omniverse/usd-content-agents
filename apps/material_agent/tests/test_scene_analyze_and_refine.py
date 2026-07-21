# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Usd, UsdGeom, UsdShade

from material_agent.scene import analyze as analyze_module
from material_agent.scene import llm_refine as llm_refine_module
from material_agent.scene.analyze import (
    _build_payload_dag,
    _collect_payload_paths_from_node,
    _count_payload_meshes,
    _detect_payload_groups,
    _detect_prototype_groups,
    _detect_structural_duplicates,
    _extract_large_payload_representatives,
    _extract_prototype,
    _extract_prototype_sources,
    analyze_scene,
)
from material_agent.scene.llm_refine import (
    _build_children_list,
    _build_split_context,
    _format_children_list,
)
from material_agent.scene.manifest import PayloadGroup, SubAsset


def _make_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def test_build_children_list_and_format(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path / "scene.usda")
    asset = UsdGeom.Xform.Define(stage, "/World/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    child_a = UsdGeom.Xform.Define(stage, "/World/Asset/A")
    mesh_a = UsdGeom.Mesh.Define(stage, "/World/Asset/A/Mesh")
    mesh_a.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    child_b = UsdGeom.Xform.Define(stage, "/World/Asset/B")
    mesh_b = UsdGeom.Mesh.Define(stage, "/World/Asset/B/Mesh")
    mesh_b.CreatePointsAttr([(0, 0, 0), (1, 1, 0)])
    UsdGeom.Xform.Define(stage, "/World/Asset/Empty")
    stage.GetRootLayer().Save()

    children = _build_children_list(stage, "/World/Asset")
    formatted = _format_children_list(children)

    assert [child["name"] for child in children] == [
        child_a.GetPrim().GetName(),
        child_b.GetPrim().GetName(),
    ]
    assert [child["mesh_count"] for child in children] == [1, 1]
    assert [child["vertex_count"] for child in children] == [3, 2]
    assert "A: 1 meshes, 3 vertices" in formatted
    assert "B: 1 meshes, 2 vertices" in formatted
    assert _build_children_list(stage, "/World/Missing") == []


def test_detect_structural_duplicates_and_count_payload_meshes(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path / "scene.usda")
    for parent_name in ("A", "B"):
        parent = UsdGeom.Xform.Define(stage, f"/World/{parent_name}")
        stage.SetDefaultPrim(parent.GetPrim())
        UsdGeom.Mesh.Define(stage, f"/World/{parent_name}/Mesh")
    container = UsdGeom.Xform.Define(stage, "/World/C")
    stage.SetDefaultPrim(container.GetPrim())
    nested = UsdGeom.Xform.Define(stage, "/World/C/Nested")
    UsdGeom.Mesh.Define(stage, f"{nested.GetPath()}/Mesh")
    stage.GetRootLayer().Save()

    sub_assets = [
        SubAsset(id="a", name="A", prim_path="/World/A"),
        SubAsset(id="b", name="B", prim_path="/World/B"),
        SubAsset(id="c", name="C", prim_path="/World/C"),
        SubAsset(
            id="skip",
            name="Skip",
            prim_path="/World/DoesNotExist",
            instance_group="native_group",
        ),
    ]

    updated_assets, groups = _detect_structural_duplicates(stage, sub_assets)

    assert updated_assets[0].instance_group is None
    assert updated_assets[1].instance_group == "structural_A"
    assert len(groups) == 1
    assert groups[0].representative_id == "a"
    assert groups[0].member_paths == ["/World/B"]

    payload_path = tmp_path / "payload.usda"
    payload_stage = _make_stage(payload_path)
    UsdGeom.Mesh.Define(payload_stage, "/World/Mesh")
    payload_stage.GetRootLayer().Save()
    empty_path = tmp_path / "empty.usda"
    empty_stage = _make_stage(empty_path)
    empty_stage.GetRootLayer().Save()

    assert _count_payload_meshes(str(payload_path)) == 1
    assert _count_payload_meshes(str(empty_path)) == 0


def test_build_payload_dag_and_detect_payload_groups(
    monkeypatch, tmp_path: Path
) -> None:
    payload_a = tmp_path / "Payload A.usda"
    payload_b = tmp_path / "nested.usda"
    for path in (payload_a, payload_b):
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "material_agent.scene.payload_dag_utils.build_dag",
        lambda roots: {
            str(payload_a.resolve()): {str(payload_b.resolve())},
            str(payload_b.resolve()): set(),
        },
    )
    monkeypatch.setattr(
        "material_agent.scene.payload_dag_utils.compute_depths",
        lambda adj: {
            str(payload_a.resolve()): 1,
            str(payload_b.resolve()): 0,
        },
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._count_payload_meshes",
        lambda payload_file: 0 if Path(payload_file).name == "nested.usda" else 3,
    )

    built = _build_payload_dag(
        [
            PayloadGroup(
                id="payload_payload_a",
                group_name="payload_a",
                payload_file=str(payload_a.resolve()),
                instance_count=2,
                instance_paths=["/World/A"],
            )
        ]
    )

    nested_group = next(
        pg for pg in built if pg.payload_file == str(payload_b.resolve())
    )
    assert nested_group.depth == 0
    assert nested_group.status == "skipped"
    assert nested_group.parent_payload_files == [str(payload_a.resolve())]

    class FakePrim:
        def __init__(self, path: str, is_instance: bool, marker: str) -> None:
            self._path = path
            self._is_instance = is_instance
            self._marker = marker

        def IsInstance(self) -> bool:
            return self._is_instance

        def GetPath(self):
            return self._path

        def GetPrimIndex(self):
            return SimpleNamespace(rootNode=self._marker)

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim("/World/A", True, "a"),
                FakePrim("/World/B", True, "b"),
                FakePrim("/World/C", False, "skip"),
            ]

    monkeypatch.setattr(
        "material_agent.scene.analyze._collect_payload_paths_from_node",
        lambda node, scene_dir: (
            [str(payload_a.resolve())] if node == "a" else [str(payload_b.resolve())]
        ),
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._build_payload_dag", lambda groups: groups
    )

    groups = _detect_payload_groups(FakeStage(), tmp_path / "scene.usda")

    assert len(groups) == 2
    groups_by_name = {group.group_name: group for group in groups}
    assert groups_by_name["nested"].status == "skipped"
    assert groups_by_name["payload_a"].instance_paths == ["/World/A"]


def test_collect_payload_paths_from_node(monkeypatch, tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_path.write_text("", encoding="utf-8")
    child_payload_path = tmp_path / "child_payload.usda"
    child_payload_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "pxr.Pcp.ArcTypePayload",
        "payload",
        raising=False,
    )

    node = SimpleNamespace(
        arcType="other",
        layerStack=SimpleNamespace(layers=[]),
        children=[
            SimpleNamespace(
                arcType="payload",
                layerStack=SimpleNamespace(
                    layers=[SimpleNamespace(realPath=str(payload_path))]
                ),
                children=[],
            ),
            SimpleNamespace(
                arcType="other",
                layerStack=SimpleNamespace(layers=[]),
                children=[
                    SimpleNamespace(
                        arcType="payload",
                        layerStack=SimpleNamespace(
                            layers=[SimpleNamespace(realPath=str(child_payload_path))]
                        ),
                        children=[],
                    )
                ],
            ),
        ],
    )

    collected = _collect_payload_paths_from_node(node, tmp_path)
    assert collected == [
        str(payload_path.resolve()),
        str(child_payload_path.resolve()),
    ]


def test_refine_objects_with_llm_handles_auto_and_llm_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    children_map = {
        "/keep-small": [
            {"name": "a", "path": "/keep-small/a", "mesh_count": 10, "vertex_count": 1},
            {"name": "b", "path": "/keep-small/b", "mesh_count": 10, "vertex_count": 1},
        ],
        "/auto-descend": [
            {
                "name": "descended",
                "path": "/auto-descend/child",
                "mesh_count": 120,
                "vertex_count": 8,
            }
        ],
        "/auto-descend/child": [],
        "/auto-split": [
            {
                "name": "one",
                "path": "/auto-split/one",
                "mesh_count": 150,
                "vertex_count": 5,
            },
            {
                "name": "two",
                "path": "/auto-split/two",
                "mesh_count": 160,
                "vertex_count": 5,
            },
            {
                "name": "leaf",
                "path": "/auto-split/leaf",
                "mesh_count": 1,
                "vertex_count": 1,
            },
        ],
        "/auto-split/one": [],
        "/auto-split/two": [],
        "/auto-keep-leaves": [
            {
                "name": "leaf-a",
                "path": "/auto-keep-leaves/a",
                "mesh_count": 1,
                "vertex_count": 1,
            },
            {
                "name": "leaf-b",
                "path": "/auto-keep-leaves/b",
                "mesh_count": 1,
                "vertex_count": 1,
            },
            {
                "name": "leaf-c",
                "path": "/auto-keep-leaves/c",
                "mesh_count": 1,
                "vertex_count": 1,
            },
        ],
        "/auto-split-large-child": [
            {
                "name": "large",
                "path": "/auto-split-large-child/large",
                "mesh_count": 150,
                "vertex_count": 10,
            },
            {
                "name": "leaf-a",
                "path": "/auto-split-large-child/a",
                "mesh_count": 1,
                "vertex_count": 1,
            },
            {
                "name": "leaf-b",
                "path": "/auto-split-large-child/b",
                "mesh_count": 1,
                "vertex_count": 1,
            },
        ],
        "/auto-split-large-child/large": [],
        "/llm-split": [
            {
                "name": "left",
                "path": "/llm-split/left",
                "mesh_count": 130,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/llm-split/right",
                "mesh_count": 140,
                "vertex_count": 5,
            },
        ],
        "/llm-split/left": [],
        "/llm-split/right": [],
        "/llm-keep": [
            {
                "name": "left",
                "path": "/llm-keep/left",
                "mesh_count": 120,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/llm-keep/right",
                "mesh_count": 125,
                "vertex_count": 5,
            },
        ],
        "/parse-fail": [
            {
                "name": "left",
                "path": "/parse-fail/left",
                "mesh_count": 120,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/parse-fail/right",
                "mesh_count": 125,
                "vertex_count": 5,
            },
        ],
    }

    monkeypatch.setattr(
        "material_agent.scene.llm_refine._build_children_list",
        lambda stage, prim_path: children_map.get(prim_path, []),
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.get_subtree_geometry_stats",
        lambda stage, path, skip_geometry=False: {
            "mesh_count": next(
                child["mesh_count"]
                for values in children_map.values()
                for child in values
                if child["path"] == path
            ),
            "vertex_count": 42,
            "face_count": 7,
            "prim_type_breakdown": {"Mesh": 1},
        },
    )

    responses = iter(
        [
            SimpleNamespace(content='{"action": "split", "reason": "modular"}'),
            SimpleNamespace(content='{"action": "keep", "reason": "single object"}'),
            SimpleNamespace(content="not json"),
        ]
    )
    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config, defaults=None: SimpleNamespace(
            invoke=lambda messages: next(responses)
        ),
    )
    monkeypatch.setattr(
        "world_understanding.utils.llm_parsing.extract_json_from_llm_response",
        lambda content, expected_keys=None: (
            json.loads(content) if content.startswith("{") else None
        ),
    )

    objects = [
        {
            "id": "obj_001",
            "name": "keep-small",
            "path": "/keep-small",
            "mesh_count": 50,
            "vertex_count": 1,
        },
        {
            "id": "obj_002",
            "name": "auto-descend",
            "path": "/auto-descend",
            "mesh_count": 200,
            "vertex_count": 1,
        },
        {
            "id": "obj_003",
            "name": "auto-split",
            "path": "/auto-split",
            "mesh_count": 220,
            "vertex_count": 1,
        },
        {
            "id": "obj_004",
            "name": "auto-keep-leaves",
            "path": "/auto-keep-leaves",
            "mesh_count": 210,
            "vertex_count": 1,
        },
        {
            "id": "obj_005",
            "name": "auto-split-large-child",
            "path": "/auto-split-large-child",
            "mesh_count": 210,
            "vertex_count": 1,
        },
        {
            "id": "obj_006",
            "name": "llm-split",
            "path": "/llm-split",
            "mesh_count": 210,
            "vertex_count": 1,
        },
        {
            "id": "obj_007",
            "name": "llm-keep",
            "path": "/llm-keep",
            "mesh_count": 210,
            "vertex_count": 1,
        },
        {
            "id": "obj_008",
            "name": "parse-fail",
            "path": "/parse-fail",
            "mesh_count": 210,
            "vertex_count": 1,
        },
    ]

    refined, instance_groups = llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=objects,
        instance_groups=[{"group_name": "native"}],
        llm_config={"backend": "mock", "model": "mock"},
        auto_split_threshold=3,
        min_mesh_for_review=100,
    )

    refined_paths = {obj["path"] for obj in refined}
    assert instance_groups == [{"group_name": "native"}]
    assert "/keep-small" in refined_paths
    assert "/auto-descend" not in refined_paths
    assert "/auto-descend/child" in refined_paths
    assert "/auto-split" not in refined_paths
    assert "/auto-split/one" in refined_paths
    assert "/auto-keep-leaves" in refined_paths
    assert "/auto-split-large-child" not in refined_paths
    assert "/auto-split-large-child/large" in refined_paths
    assert "/llm-split" not in refined_paths
    assert "/llm-split/left" in refined_paths
    assert "/llm-keep" in refined_paths
    assert "/parse-fail" in refined_paths

    descended = next(obj for obj in refined if obj["path"] == "/auto-descend/child")
    assert descended["split_context"] is None
    split_child = next(obj for obj in refined if obj["path"] == "/llm-split/left")
    assert split_child["split_context"]["parent_name"] == "llm-split"
    assert split_child["split_context"]["sibling_names"] == ["left", "right"]

    skipped_refine = llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=objects,
        instance_groups=[],
        llm_config={"backend": "mock", "model": "mock"},
        auto_split_threshold=3,
        min_mesh_for_review=100,
    )
    assert skipped_refine[0]


def test_refine_objects_exits_when_no_llm_or_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = [
        {
            "id": "obj_bad",
            "name": "plain",
            "path": "/plain",
            "mesh_count": 10,
            "vertex_count": 1,
        }
    ]

    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config, defaults=None: None,
    )

    assert llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=objects,
        instance_groups=[{"group_name": "native"}],
        llm_config={"backend": "mock"},
    ) == (objects, [{"group_name": "native"}])

    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config, defaults=None: object(),
    )
    monkeypatch.setattr(
        "material_agent.scene.llm_refine._build_children_list",
        lambda stage, prim_path: [],
    )

    assert llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=objects,
        instance_groups=[],
        llm_config={"backend": "mock"},
    ) == (objects, [])


def test_refine_objects_keeps_candidates_at_depth_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = [
        {
            "id": "obj_001",
            "name": "depth-limit",
            "path": "/depth-limit",
            "mesh_count": 200,
            "vertex_count": 1,
        }
    ]
    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config, defaults=None: object(),
    )
    monkeypatch.setattr(
        "material_agent.scene.llm_refine._build_children_list",
        lambda stage, prim_path: [
            {
                "name": "left",
                "path": "/depth-limit/left",
                "mesh_count": 120,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/depth-limit/right",
                "mesh_count": 125,
                "vertex_count": 5,
            },
        ],
    )

    refined, _ = llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=objects,
        instance_groups=[],
        llm_config={"backend": "mock"},
        max_split_depth=0,
        min_mesh_for_review=100,
    )

    assert refined == objects


def test_refine_objects_keeps_llm_response_when_usage_tracking_fails(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    children_map = {
        "/llm-split": [
            {
                "name": "left",
                "path": "/llm-split/left",
                "mesh_count": 120,
                "vertex_count": 5,
            },
            {
                "name": "right",
                "path": "/llm-split/right",
                "mesh_count": 125,
                "vertex_count": 5,
            },
        ]
    }

    monkeypatch.setattr(
        "material_agent.scene.llm_refine._build_children_list",
        lambda stage, prim_path: children_map.get(prim_path, []),
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.get_subtree_geometry_stats",
        lambda stage, path, skip_geometry=False: {
            "mesh_count": next(
                child["mesh_count"]
                for values in children_map.values()
                for child in values
                if child["path"] == path
            ),
            "vertex_count": 42,
            "face_count": 7,
            "prim_type_breakdown": {"Mesh": 1},
        },
    )
    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config, defaults=None: SimpleNamespace(
            invoke=lambda messages: SimpleNamespace(
                content='{"action": "split", "reason": "modular"}'
            )
        ),
    )
    monkeypatch.setattr(
        "world_understanding.utils.llm_parsing.extract_json_from_llm_response",
        lambda content, expected_keys=None: json.loads(content),
    )

    def fail_usage_recording(*args, **kwargs):
        raise RuntimeError("usage tracker failed")

    monkeypatch.setattr(
        "material_agent.scene.stats.record_model_response_usage",
        fail_usage_recording,
    )

    caplog.set_level(logging.WARNING, logger="material_agent.scene.llm_refine")
    refined, _ = llm_refine_module.refine_objects_with_llm(
        stage=object(),
        objects=[
            {
                "id": "obj_001",
                "name": "llm-split",
                "path": "/llm-split",
                "mesh_count": 210,
                "vertex_count": 1,
            }
        ],
        instance_groups=[],
        llm_config={"backend": "mock", "model": "mock", "max_workers": 1},
        min_mesh_for_review=100,
        token_tracker=object(),
    )

    refined_paths = {obj["path"] for obj in refined}
    assert "/llm-split" not in refined_paths
    assert "/llm-split/left" in refined_paths
    assert "Failed to record LLM usage" in caplog.text


def test_split_context_and_analyze_scene_main(monkeypatch, tmp_path: Path) -> None:
    parent_context = _build_split_context(
        {"name": "Parent", "split_context": {"ancestors": ["Root"]}},
        "Child",
        ["Sibling"],
    )
    assert parent_context == {
        "parent_name": "Parent",
        "sibling_names": ["Sibling"],
        "ancestors": ["Root", "Parent"],
    }

    fake_stage = object()
    monkeypatch.setattr("pxr.Usd.Stage.Open", lambda path: fake_stage)
    monkeypatch.setattr(
        "world_understanding.utils.usd.composition.collect_composition_arcs",
        lambda stage: {
            "sublayer_count": 1,
            "reference_count": 2,
            "unique_sub_usd_count": 3,
        },
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.collect_mesh_geometry_stats",
        lambda stage, skip_geometry=False: {
            "total_prims": 20,
            "total_meshes": 10,
            "total_vertices": 100,
        },
    )

    objects = [
        {
            "id": "obj_keep",
            "name": "Keep",
            "path": "/World/Keep",
            "mesh_count": 10,
            "vertex_count": 20,
        },
        {
            "id": "obj_skip",
            "name": "Skip",
            "path": "/World/Skip",
            "mesh_count": 10,
            "vertex_count": 20,
        },
        {
            "id": "obj_small",
            "name": "Small",
            "path": "/World/Small",
            "mesh_count": 1,
            "vertex_count": 5,
        },
        {
            "id": "obj_outside",
            "name": "Outside",
            "path": "/Other/Outside",
            "mesh_count": 12,
            "vertex_count": 24,
        },
        {
            "id": "obj_child",
            "name": "Child",
            "path": "/World/Parent/Child",
            "mesh_count": 12,
            "vertex_count": 24,
            "instance_group": None,
        },
    ]
    instance_groups_raw = [
        {
            "group_name": "direct_group",
            "source_file": "/tmp/direct.usd",
            "instance_count": 1,
            "member_paths": ["/World/Keep"],
        },
        {
            "group_name": "native_group",
            "source_file": "/tmp/source.usd",
            "instance_count": 2,
            "member_paths": ["/World/Parent"],
        },
    ]
    monkeypatch.setattr(
        "world_understanding.functions.graphics.usd_scene_analysis.detect_objects",
        lambda *args, **kwargs: (objects, instance_groups_raw),
    )
    monkeypatch.setattr(
        "material_agent.scene.llm_refine.refine_objects_with_llm",
        lambda **kwargs: (objects, instance_groups_raw),
    )

    def fake_detect_dupes(stage, sub_assets):
        for sub_asset in sub_assets:
            if sub_asset.id == "obj_child":
                sub_asset.instance_group = "structural_dup"
        return sub_assets, []

    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_structural_duplicates", fake_detect_dupes
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_payload_groups",
        lambda stage, scene_usd_path: [
            PayloadGroup(
                id="payload_one",
                group_name="payload_one",
                payload_file="/tmp/payload_one.usd",
                instance_count=2,
                instance_paths=["/World/InstanceA"],
            )
        ],
    )
    captured_working_dirs = {}

    def fake_extract_representatives(
        payload_groups,
        scene_usd_path,
        working_dir=None,
    ):
        captured_working_dirs["extract"] = working_dir
        payload_groups[0] = PayloadGroup(
            **{
                **payload_groups[0].__dict__,
                "representative_path": "/tmp/payload_one_representative.usd",
            }
        )

    def fake_detect_prototypes(stage, scene_usd_path, working_dir=None):
        captured_working_dirs["prototype"] = working_dir
        return [
            PayloadGroup(
                id="proto_one",
                group_name="proto_one",
                payload_file="/tmp/proto_one.usd",
                instance_count=1,
                instance_paths=["/World/Proto"],
            )
        ]

    monkeypatch.setattr(
        "material_agent.scene.analyze._extract_large_payload_representatives",
        fake_extract_representatives,
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_prototype_groups",
        fake_detect_prototypes,
    )

    analysis_dir = tmp_path / "analysis"
    manifest = analyze_scene(
        tmp_path / "scene.usda",
        filters={
            "include_paths": ["/World"],
            "exclude_paths": ["/World/Skip"],
            "min_mesh_count": 5,
            "detect_structural_duplicates": True,
        },
        llm_config={"backend": "mock", "model": "mock"},
        working_dir=analysis_dir,
    )

    assert [sa.id for sa in manifest.sub_assets] == ["obj_keep", "obj_child"]
    assert manifest.sub_assets[1].instance_group is None
    representatives = {
        group.group_name: group.representative_id for group in manifest.instance_groups
    }
    assert representatives["direct_group"] == "obj_keep"
    assert representatives["native_group"] == "obj_child"
    assert manifest.analysis["total_objects_detected"] == 5
    assert manifest.analysis["total_objects_after_filter"] == 2
    assert manifest.analysis["total_payload_groups"] == 2
    assert manifest.payload_groups[0].group_name == "payload_one"
    assert manifest.payload_groups[1].group_name == "proto_one"
    assert captured_working_dirs == {
        "extract": analysis_dir,
        "prototype": analysis_dir,
    }


@pytest.mark.parametrize(
    "filter_key",
    ["exclude_invisible_assets", "skip_invisible"],
)
def test_analyze_scene_skip_invisible_traverses_instance_proxies(
    monkeypatch, tmp_path: Path, filter_key: str
) -> None:
    scene_path = tmp_path / "instanced_scene.usda"
    stage = _make_stage(scene_path)
    prototype = UsdGeom.Xform.Define(stage, "/World/Prototype")
    UsdGeom.Mesh.Define(stage, f"{prototype.GetPath()}/Mesh")

    instance = UsdGeom.Xform.Define(stage, "/World/InstancedAsset").GetPrim()
    instance.GetReferences().AddInternalReference(str(prototype.GetPath()))
    instance.SetInstanceable(True)

    material_asset = UsdGeom.Xform.Define(stage, "/World/AssetWithMaterial")
    UsdShade.Material.Define(stage, f"{material_asset.GetPath()}/Looks/Material")
    UsdGeom.Mesh.Define(stage, f"{material_asset.GetPath()}/Mesh")

    hidden_group = UsdGeom.Xform.Define(stage, "/World/HiddenGroup")
    UsdGeom.Imageable(hidden_group).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    hidden_asset = UsdGeom.Xform.Define(stage, f"{hidden_group.GetPath()}/HiddenAsset")
    UsdGeom.Mesh.Define(stage, f"{hidden_asset.GetPath()}/Mesh")
    hidden_child_container = UsdGeom.Xform.Define(
        stage,
        "/World/HiddenChildContainer",
    )
    hidden_child = UsdGeom.Xform.Define(
        stage,
        f"{hidden_child_container.GetPath()}/HiddenChild",
    )
    UsdGeom.Imageable(hidden_child).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdGeom.Mesh.Define(stage, f"{hidden_child.GetPath()}/Mesh")
    stage.GetRootLayer().Save()

    objects = [
        {
            "id": "obj_instance",
            "name": "InstancedAsset",
            "path": str(instance.GetPath()),
            "mesh_count": 1,
            "vertex_count": 0,
        },
        {
            "id": "obj_material",
            "name": "AssetWithMaterial",
            "path": str(material_asset.GetPath()),
            "mesh_count": 1,
            "vertex_count": 0,
        },
        {
            "id": "obj_hidden",
            "name": "HiddenAsset",
            "path": str(hidden_asset.GetPath()),
            "mesh_count": 1,
            "vertex_count": 0,
        },
        {
            "id": "obj_hidden_child",
            "name": "HiddenChildContainer",
            "path": str(hidden_child_container.GetPath()),
            "mesh_count": 1,
            "vertex_count": 0,
        },
        {
            "id": "obj_missing",
            "name": "MissingAsset",
            "path": "/World/MissingAsset",
            "mesh_count": 1,
            "vertex_count": 0,
        },
    ]
    monkeypatch.setattr(
        "world_understanding.functions.graphics.usd_scene_analysis.detect_objects",
        lambda *args, **kwargs: (objects, []),
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_payload_groups",
        lambda stage, scene_usd_path: [],
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._detect_prototype_groups",
        lambda stage, scene_usd_path: [],
    )

    manifest = analyze_scene(
        scene_path,
        filters={filter_key: True},
        llm_config=None,
    )

    assert [sub_asset.id for sub_asset in manifest.sub_assets] == [
        "obj_instance",
        "obj_material",
    ]


def test_scene_analyze_payload_and_structural_edge_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = _make_stage(tmp_path / "scene.usda")
    UsdGeom.Mesh.Define(stage, "/World/A/Mesh")
    UsdGeom.Mesh.Define(stage, "/World/B/Mesh")
    stage.GetRootLayer().Save()

    sub_assets = [
        SubAsset(id="a", name="A", prim_path="/World/A"),
        SubAsset(id="b", name="B", prim_path="/World/B"),
        SubAsset(id="missing", name="Missing", prim_path="/World/Missing"),
    ]
    updated, groups = _detect_structural_duplicates(stage, sub_assets)
    assert updated[1].instance_group == "structural_A"
    assert updated[2].instance_group is None
    assert groups[0].member_paths == ["/World/B"]

    monkeypatch.setattr("pxr.Usd.Stage.Open", lambda payload_file: None)
    assert _count_payload_meshes(str(tmp_path / "missing.usda")) == 0

    def raise_open(payload_file: str):
        raise RuntimeError("boom")

    monkeypatch.setattr("pxr.Usd.Stage.Open", raise_open)
    assert _count_payload_meshes(str(tmp_path / "broken.usda")) == 0

    payload_path = tmp_path / "Payload A.usda"
    payload_path.write_text("", encoding="utf-8")

    class FakePrim:
        def __init__(
            self, path: str, is_instance: bool, root_node: object | None
        ) -> None:
            self._path = path
            self._is_instance = is_instance
            self._root_node = root_node

        def IsInstance(self) -> bool:
            return self._is_instance

        def GetPath(self) -> str:
            return self._path

        def GetPrimIndex(self):
            if self._root_node == "no-index":
                return None
            return SimpleNamespace(rootNode=self._root_node)

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim("/World/NoIndex", True, "no-index"),
                FakePrim("/World/NoRoot", True, None),
                FakePrim("/World/Instance", True, "payload"),
                FakePrim("/World/Plain", False, "ignored"),
            ]

    monkeypatch.setattr(
        "material_agent.scene.analyze._collect_payload_paths_from_node",
        lambda node, scene_dir: [str(payload_path.resolve())],
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._count_payload_meshes", lambda payload_file: 1
    )
    monkeypatch.setattr(
        "material_agent.scene.analyze._build_payload_dag", lambda groups: groups
    )

    groups = _detect_payload_groups(FakeStage(), tmp_path / "scene.usda")
    assert len(groups) == 1
    assert groups[0].instance_paths == ["/World/Instance"]


def test_scene_analyze_prototype_group_edge_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakePrim:
        def __init__(
            self,
            path: str,
            *,
            name: str | None = None,
            is_instance: bool = False,
            prototype: object | None = None,
            valid: bool = True,
            is_mesh: bool = False,
        ) -> None:
            self._path = path
            self._name = name or path.rsplit("/", 1)[-1]
            self._is_instance = is_instance
            self._prototype = prototype
            self._valid = valid
            self._is_mesh = is_mesh

        def GetName(self) -> str:
            return self._name

        def GetPath(self) -> str:
            return self._path

        def GetPrototype(self):
            return self._prototype

        def IsInstance(self) -> bool:
            return self._is_instance

        def IsValid(self) -> bool:
            return self._valid

        def IsA(self, schema) -> bool:
            return self._is_mesh

    empty_proto = FakePrim("/__Prototype_1")
    mesh_without_instance = FakePrim("/__Prototype_2")
    mesh_with_instance = FakePrim("/__Prototype_3")
    instance = FakePrim(
        "/World/Instance",
        name="Nice Instance",
        is_instance=True,
        prototype=mesh_with_instance,
    )
    invalid_named_instance = FakePrim(
        "/World/InvalidName",
        is_instance=True,
        prototype=mesh_with_instance,
        valid=False,
    )

    class FakeStage:
        def GetPrototypes(self):
            return [empty_proto, mesh_without_instance, mesh_with_instance]

        def Traverse(self):
            return [instance, invalid_named_instance]

        def GetPrimAtPath(self, path: str):
            if path == "/World/Instance":
                return instance
            return invalid_named_instance

    def fake_prim_range(proto, *args):
        if proto is empty_proto:
            return [FakePrim("/__Prototype_1/Xform")]
        return [FakePrim(str(proto.GetPath()) + "/Mesh", is_mesh=True)]

    monkeypatch.setattr("pxr.Usd.PrimRange", fake_prim_range)
    monkeypatch.setattr("pxr.Usd.TraverseInstanceProxies", lambda: object())

    def fail_extract(stage, representative_path: str, output_path: str) -> None:
        raise RuntimeError("extract failed")

    monkeypatch.setattr("material_agent.scene.analyze._extract_prototype", fail_extract)

    assert _detect_prototype_groups(FakeStage(), tmp_path / "scene.usda") == []

    empty_stage = SimpleNamespace(GetPrototypes=lambda: [])
    assert _detect_prototype_groups(empty_stage, tmp_path / "scene.usda") == []

    def write_extract(stage, representative_path: str, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(
        "material_agent.scene.analyze._extract_prototype", write_extract
    )

    active = _detect_prototype_groups(FakeStage(), tmp_path / "scene.usda")
    assert len(active) == 1
    assert active[0].group_name == "nice_instance"
    assert active[0].payload_file.endswith("nice_instance.usd")


def test_extract_prototype_mask_paths_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    masks: list[list[str]] = []
    exports: list[str] = []

    class FakeProto:
        def GetPath(self) -> str:
            return "/__Prototype_1"

    class FakePrim:
        def __init__(self, *, is_instance: bool, prototype: object | None) -> None:
            self._is_instance = is_instance
            self._prototype = prototype
            self.instanceable: bool | None = None

        def IsInstance(self) -> bool:
            return self._is_instance

        def GetPrototype(self):
            return self._prototype

        def SetInstanceable(self, value: bool) -> None:
            self.instanceable = value

    class FakeFlatLayer:
        def Export(self, path: str) -> None:
            exports.append(path)

    class FakeMaskedStage:
        def __init__(self) -> None:
            self.masked_prim = FakePrim(is_instance=True, prototype=FakeProto())

        def GetPrimAtPath(self, path: str):
            return self.masked_prim

        def Flatten(self) -> FakeFlatLayer:
            return FakeFlatLayer()

    class FakeStage:
        def GetRootLayer(self):
            return object()

        def GetPrimAtPath(self, path: str):
            return FakePrim(is_instance=True, prototype=FakeProto())

    def fake_mask(paths: list[str]):
        masks.append(list(paths))
        return object()

    monkeypatch.setattr("pxr.Usd.StagePopulationMask", fake_mask)
    monkeypatch.setattr("pxr.Usd.Stage.OpenMasked", lambda *args: FakeMaskedStage())

    out_file = tmp_path / "prototype.usda"
    _extract_prototype(FakeStage(), "/World/Instance", str(out_file))
    assert masks == [["/World/Instance", "/__Prototype_1"]]
    assert exports == [str(out_file)]

    monkeypatch.setattr("pxr.Usd.Stage.OpenMasked", lambda *args: None)
    with pytest.raises(RuntimeError, match="Failed to open masked stage"):
        _extract_prototype(FakeStage(), "/World/Instance", str(out_file))


def test_large_payload_representative_extraction_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    skipped = PayloadGroup(
        id="skipped",
        group_name="Skipped",
        payload_file=str(tmp_path / "skipped.usda"),
        status="skipped",
    )
    missing = PayloadGroup(id="missing", group_name="Missing", payload_file="")
    unreadable = PayloadGroup(
        id="unreadable",
        group_name="Unreadable",
        payload_file=str(tmp_path / "unreadable.usda"),
    )
    small = PayloadGroup(
        id="small",
        group_name="Small",
        payload_file=str(tmp_path / "small.usda"),
    )
    large_without_instances = PayloadGroup(
        id="plain",
        group_name="Plain",
        payload_file=str(tmp_path / "plain.usda"),
    )
    large_with_instances = PayloadGroup(
        id="large",
        group_name="Large",
        payload_file=str(tmp_path / "large.usda"),
    )
    large_with_working_dir = PayloadGroup(
        id="working",
        group_name="WorkingDir",
        payload_file=str(tmp_path / "working.usda"),
    )
    representative = tmp_path / "large_representative.usda"
    working_representative = tmp_path / "working_representative.usda"

    def fake_getsize(path) -> int:
        path = str(path)
        if path.endswith("unreadable.usda"):
            raise OSError("missing")
        if path in {str(representative), str(working_representative)}:
            return 2 * 1024 * 1024
        if path.endswith(("plain.usda", "large.usda", "working.usda")):
            return analyze_module._LARGE_PAYLOAD_THRESHOLD_BYTES + 1024
        return 1

    analysis_dir = tmp_path / "analysis"
    seen_working_dirs = []

    def fake_extract(
        payload_file: str,
        scene_usd_path: Path,
        working_dir: Path | None = None,
    ) -> Path | None:
        seen_working_dirs.append(working_dir)
        if payload_file.endswith("working.usda"):
            assert working_dir == analysis_dir
            return working_representative
        assert working_dir is None
        if payload_file.endswith("large.usda"):
            return representative
        return None

    monkeypatch.setattr("os.path.getsize", fake_getsize)
    monkeypatch.setattr(
        "material_agent.scene.analyze._extract_prototype_sources", fake_extract
    )

    _extract_large_payload_representatives(
        [
            skipped,
            missing,
            unreadable,
            small,
            large_without_instances,
            large_with_instances,
        ],
        tmp_path / "scene.usda",
    )
    _extract_large_payload_representatives(
        [large_with_working_dir],
        tmp_path / "scene.usda",
        working_dir=analysis_dir,
    )

    assert large_without_instances.representative_path is None
    assert large_with_instances.representative_path == str(representative)
    assert large_with_working_dir.representative_path == str(working_representative)
    assert seen_working_dirs == [None, None, analysis_dir]


def test_extract_prototype_sources_failures_and_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload_path = tmp_path / "payload.usda"
    scene_path = tmp_path / "scene.usda"

    def raise_open(payload_file: str):
        raise RuntimeError("cannot open")

    monkeypatch.setattr("pxr.Usd.Stage.Open", raise_open)
    assert _extract_prototype_sources(str(payload_path), scene_path) is None

    monkeypatch.setattr("pxr.Usd.Stage.Open", lambda payload_file: None)
    assert _extract_prototype_sources(str(payload_path), scene_path) is None

    class EmptyPrototypeStage:
        def GetPrototypes(self):
            return []

    monkeypatch.setattr(
        "pxr.Usd.Stage.Open", lambda payload_file: EmptyPrototypeStage()
    )
    assert _extract_prototype_sources(str(payload_path), scene_path) is None

    class NoSourceStage:
        def GetPrototypes(self):
            return [object()]

        def GetPseudoRoot(self):
            return SimpleNamespace(GetChildren=lambda: [])

    monkeypatch.setattr("pxr.Usd.Stage.Open", lambda payload_file: NoSourceStage())
    assert _extract_prototype_sources(str(payload_path), scene_path) is None

    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    UsdGeom.Xform.Define(payload_stage, "/World")
    source = UsdGeom.Xform.Define(payload_stage, "/World/Source").GetPrim()
    UsdGeom.Mesh.Define(payload_stage, "/World/Source/Mesh")
    instance = UsdGeom.Xform.Define(payload_stage, "/World/Instance").GetPrim()
    instance.GetReferences().AddInternalReference(str(source.GetPath()))
    instance.SetInstanceable(True)
    payload_stage.GetRootLayer().Save()

    monkeypatch.undo()
    extracted = _extract_prototype_sources(str(payload_path), scene_path)
    assert extracted == tmp_path / ".scene_working" / "representatives" / (
        "payload_representative.usd"
    )
    assert extracted.exists()

    monkeypatch.setattr("pxr.Usd.Stage.Open", lambda payload_file: payload_stage)
    monkeypatch.setattr("pxr.Usd.Stage.OpenMasked", lambda *args: None)
    assert _extract_prototype_sources(str(payload_path), scene_path) is None
