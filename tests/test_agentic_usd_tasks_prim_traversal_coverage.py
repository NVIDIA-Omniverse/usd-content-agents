# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for deterministic prim traversal helpers."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import world_understanding.agentic.usd_tasks.prim_traversal as prim_traversal
from world_understanding.agentic.usd_tasks.prim_traversal import (
    USDPrimTraversalAndRenderingTask,
    _blank_dataset_render_message,
    _is_blank_render_status,
    _is_config_enabled,
    _resolve_render_endpoint_for_diagnostic,
    _sanitize_render_endpoint_for_diagnostic,
    _validate_positive_int_config,
    _zero_image_render_error_message,
    compute_relative_metrics,
    get_stage_world_bbox,
    get_world_bbox_from_prim,
    prim_path_to_directory_structure,
    scale_bbox_by_mpu,
)
from world_understanding.functions.graphics.rendering import (
    RemoteRenderingBackend,
    RenderingConfig,
)


def _stage_with_mesh() -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Cube")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(0, 1, 0),
                Gf.Vec3f(0, 0, 1),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 1)]))
    return stage, mesh.GetPrim()


def _bind_mdl_material(stage: Usd.Stage, prim: Usd.Prim) -> UsdShade.Material:
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Mat")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Mat/Shader")
    shader.CreateIdAttr("mdlMaterial")
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
    ).Set("OmniPBR")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return material


def test_config_endpoint_and_bbox_helpers_cover_edges(monkeypatch: pytest.MonkeyPatch):
    assert (
        _sanitize_render_endpoint_for_diagnostic(
            "https://user:secret@example.test/render?token=abc"
        )
        == "https://example.test/render"
    )
    assert _sanitize_render_endpoint_for_diagnostic("host.test:8000/path?x=1") == (
        "http://host.test:8000/path"
    )
    assert _sanitize_render_endpoint_for_diagnostic("http://[::1") == "http://[::1"

    assert _is_config_enabled(None, True) is True
    assert _is_config_enabled(" yes ", False) is True
    assert _is_config_enabled("off", True) is False
    assert _is_config_enabled("", True) is False
    assert _is_config_enabled(1, False) is True

    stage, prim = _stage_with_mesh()
    prim_bbox = get_world_bbox_from_prim(prim)
    stage_bbox = get_stage_world_bbox(stage)
    assert prim_bbox is not None
    assert stage_bbox is not None
    assert scale_bbox_by_mpu(prim_bbox, 2.0)["size"] == [2.0, 2.0, 2.0]
    assert scale_bbox_by_mpu({}, 1.0) is None
    assert scale_bbox_by_mpu(prim_bbox, 0) is None

    relative = compute_relative_metrics(prim_bbox, stage_bbox)
    assert relative is not None
    assert "relative_volume" in relative
    zero_stage = {
        "center": [0, 0, 0],
        "size": [0, 0, 0],
        "min": [0, 0, 0],
        "max": [0, 0, 0],
    }
    assert compute_relative_metrics(prim_bbox, zero_stage)["relative_volume"] == 0
    assert compute_relative_metrics(None, stage_bbox) is None

    render_cube = UsdGeom.Cube.Define(stage, "/World/RenderOnly")
    render_cube.CreatePurposeAttr(UsdGeom.Tokens.render)
    render_cube.AddTranslateOp().Set((100.0, 0.0, 0.0))
    assert get_world_bbox_from_prim(render_cube.GetPrim()) is None
    explicit_render_bbox = get_world_bbox_from_prim(
        render_cube.GetPrim(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    assert explicit_render_bbox is not None
    assert explicit_render_bbox["max"][0] == pytest.approx(101.0)
    assert get_stage_world_bbox(stage)["max"][0] == pytest.approx(1.0)
    assert get_stage_world_bbox(
        stage,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )["max"][0] == pytest.approx(101.0)

    monkeypatch.setattr(
        prim_traversal.UsdGeom.BBoxCache,
        "ComputeWorldBound",
        Mock(side_effect=RuntimeError("bbox bad")),
    )
    assert get_world_bbox_from_prim(prim) is None
    assert get_stage_world_bbox(stage) is None


def test_get_world_bbox_supports_time_and_extents_hint_controls() -> None:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    Usd.ModelAPI(world.GetPrim()).SetKind(Kind.Tokens.group)
    model = UsdGeom.Xform.Define(stage, "/World/Model")
    Usd.ModelAPI(model.GetPrim()).SetKind(Kind.Tokens.component)
    cube = UsdGeom.Cube.Define(stage, "/World/Model/Cube")
    cube.GetSizeAttr().Set(20.0)
    cube.GetSizeAttr().Set(2.0, Usd.TimeCode(0))
    extents_hint: list[Gf.Vec3d] = []
    for _purpose in UsdGeom.Imageable.GetOrderedPurposeTokens():
        extents_hint.extend(
            [
                Gf.Vec3d(-50.0, -50.0, -50.0),
                Gf.Vec3d(50.0, 50.0, 50.0),
            ]
        )
    assert UsdGeom.ModelAPI(model).SetExtentsHint(extents_hint)

    default_bbox = get_world_bbox_from_prim(model.GetPrim())
    default_without_hint = get_world_bbox_from_prim(
        model.GetPrim(),
        time_code=Usd.TimeCode.Default(),
        use_extents_hint=False,
    )
    default_with_hint = get_world_bbox_from_prim(
        model.GetPrim(),
        time_code=Usd.TimeCode.Default(),
        use_extents_hint=True,
    )
    selection_bbox = get_world_bbox_from_prim(
        model.GetPrim(),
        time_code=Usd.TimeCode(0),
        use_extents_hint=False,
    )

    assert default_bbox is not None
    assert default_without_hint is not None
    assert default_with_hint is not None
    assert selection_bbox is not None
    assert default_bbox["size"] == [20.0, 20.0, 20.0]
    assert default_without_hint["size"] == [20.0, 20.0, 20.0]
    assert default_with_hint["size"] == [100.0, 100.0, 100.0]
    assert selection_bbox["size"] == [2.0, 2.0, 2.0]


def test_explicit_members_world_bbox_rejects_invalid_members() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdShade.Material.Define(stage, "/World/Material")

    with pytest.raises(ValueError, match="Explicit bbox member does not exist"):
        prim_traversal._explicit_members_world_bbox(
            stage,
            ("/World/Missing",),
            ("default",),
        )
    with pytest.raises(
        ValueError,
        match="Explicit bbox member is not UsdGeom.Imageable",
    ):
        prim_traversal._explicit_members_world_bbox(
            stage,
            ("/World/Material",),
            ("default",),
        )


def test_endpoint_diagnostic_and_config_helper_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _sanitize_render_endpoint_for_diagnostic("[::1") == "[::1"
    assert _sanitize_render_endpoint_for_diagnostic("relative/path") == (
        "http://relative/path"
    )
    assert _sanitize_render_endpoint_for_diagnostic("/relative/path") == (
        "/relative/path"
    )
    assert _sanitize_render_endpoint_for_diagnostic("http:///missing-host") == (
        "http://http:///missing-host"
    )

    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    remote = RemoteRenderingBackend(base_url=None)
    assert _resolve_render_endpoint_for_diagnostic(remote) == (None, None, False)
    assert _zero_image_render_error_message(remote).startswith(
        "Rendering produced 0 images. Check the configured render endpoint"
    )
    assert _resolve_render_endpoint_for_diagnostic(object()) == (None, None, False)

    monkeypatch.setenv("RENDER_ENDPOINT", "   ")
    assert _resolve_render_endpoint_for_diagnostic(remote) == (
        None,
        "RENDER_ENDPOINT",
        False,
    )
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.setenv("NVCF_RENDER_FUNCTION_ID", "render-fn")
    assert _resolve_render_endpoint_for_diagnostic(remote)[1] == (
        "NVCF_RENDER_FUNCTION_ID"
    )

    nvcf_remote = RemoteRenderingBackend(base_url="abc123")
    monkeypatch.setattr(
        prim_traversal,
        "resolve_endpoint_or_function_id",
        lambda endpoint: f"https://{prim_traversal.NVCF_INVOCATION_HOST}/v2/{endpoint}",
    )
    endpoint, source, is_nvcf = _resolve_render_endpoint_for_diagnostic(nvcf_remote)
    assert source == "base_url"
    assert is_nvcf is True
    assert endpoint.endswith("/v2/abc123")
    assert "NVCF render function" in _zero_image_render_error_message(nvcf_remote)

    assert _validate_positive_int_config("batch_size", "3") == 3
    for value in (True, 0, "not-int", None):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            _validate_positive_int_config("batch_size", value)

    assert _blank_dataset_render_message(2, 3).startswith("2/3 dataset renders")
    assert _is_blank_render_status("blank_render") is True
    assert _is_blank_render_status(object()) is False
    assert _is_config_enabled(True, False) is True


def test_path_mapping_and_bbox_empty_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping_file = tmp_path / "path_mapping.json"
    mapping_file.write_text("{not-json", encoding="utf-8")
    prim_traversal._record_path_mapping(tmp_path, "original", "truncated")
    assert json.loads(mapping_file.read_text()) == {"truncated": "original"}

    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("readonly")),
    )
    prim_traversal._record_path_mapping(tmp_path, "ignored", "ignored")

    single_level = prim_path_to_directory_structure("/Cube", tmp_path, "view.png")
    assert single_level == tmp_path / "view.png"

    empty_stage = Usd.Stage.CreateInMemory()
    assert get_stage_world_bbox(empty_stage) is None
    assert get_world_bbox_from_prim(empty_stage.DefinePrim("/Empty", "Xform")) is None


def test_process_sensor_array_variants() -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()

    depth = np.array([0.0, 1.0, np.inf, 3.0], dtype=float)
    assert (
        task._process_sensor_array(depth, "depth", 2, 2, "/World/Cube", listener).mode
        == "L"
    )
    flat_rgb = np.arange(12, dtype=np.uint8)
    assert task._process_sensor_array(
        flat_rgb, "rgb", 2, 2, "/World/Cube", listener
    ).size == (2, 2)
    assert (
        task._process_sensor_array(
            np.array([1, 2, 3, 4, 5], dtype=np.uint8),
            "rgb",
            2,
            2,
            "/World/Cube",
            listener,
        )
        is None
    )
    assert (
        task._process_sensor_array(
            np.zeros((1, 1, 1, 1), dtype=np.uint8),
            "rgb",
            1,
            1,
            "/World/Cube",
            listener,
        )
        is None
    )

    seg_2d = np.array([[0, 1], [2, 2]], dtype=np.uint32)
    assert (
        task._process_sensor_array(
            seg_2d, "instance_id_segmentation", 2, 2, "/World/Cube", listener
        ).mode
        == "RGB"
    )
    seg_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    assert (
        task._process_sensor_array(
            seg_rgb, "instance_id_segmentation", 2, 2, "/World/Cube", listener
        ).mode
        == "RGB"
    )
    seg_rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    assert (
        task._process_sensor_array(
            seg_rgba, "instance_id_segmentation", 2, 2, "/World/Cube", listener
        ).mode
        == "RGB"
    )
    assert (
        task._process_sensor_array(
            np.zeros((2, 2, 2), dtype=np.uint8),
            "instance_id_segmentation",
            2,
            2,
            "/World/Cube",
            listener,
        )
        is None
    )
    assert task._process_sensor_array(
        np.arange(6, dtype=np.uint8),
        "rgb",
        3,
        2,
        "/World/Cube",
        listener,
    ).size == (3, 2)
    assert (
        task._process_sensor_array(
            np.ones((2, 2), dtype=np.float32),
            "linear_depth",
            2,
            2,
            "/World/Cube",
            listener,
        ).mode
        == "L"
    )
    assert (
        task._process_sensor_array(
            np.full((2, 2), np.inf, dtype=np.float32),
            "depth",
            2,
            2,
            "/World/Cube",
            listener,
        ).mode
        == "L"
    )
    assert (
        task._process_sensor_array(
            np.array(5, dtype=np.uint32),
            "instance_id_segmentation",
            1,
            1,
            "/World/Cube",
            listener,
        )
        is None
    )
    assert (
        task._process_sensor_array(
            np.array([object()], dtype=object),
            "custom_sensor",
            1,
            1,
            "/World/Cube",
            listener,
        )
        is None
    )


def test_existing_file_helpers_record_rgb_and_sensors(tmp_path: Path) -> None:
    task = USDPrimTraversalAndRenderingTask()
    config = RenderingConfig(camera_ordering=["+x"], camera_name_prefix="Cam")
    prim_path = "/World/Cube"

    assert not task._check_prim_files_exist(
        prim_path,
        tmp_path,
        config,
        rendering_modes=["prim_only"],
        sensor_modes=["depth"],
    )

    rgb = prim_traversal.prim_path_to_directory_structure(
        prim_path, tmp_path, "Cube_posx_prim_only.png"
    )
    sensor = prim_traversal.prim_path_to_directory_structure(
        prim_path, tmp_path, "Cube_posx_depth.png"
    )
    Image.new("RGB", (1, 1), "red").save(rgb)
    Image.new("L", (1, 1), 0).save(sensor)
    for filename in (
        "Cube_posx_prim_with_stage.png",
        "Cube_posx_composition.png",
        "Cube_posx_custom.png",
    ):
        Image.new("RGB", (1, 1), "blue").save(
            prim_traversal.prim_path_to_directory_structure(
                prim_path,
                tmp_path,
                filename,
            )
        )

    assert task._check_prim_files_exist(
        prim_path,
        tmp_path,
        config,
        rendering_modes=["prim_only"],
        sensor_modes=["depth"],
    )
    assert task._check_prim_files_exist(prim_path, tmp_path, config)
    assert task._check_prim_files_exist(
        prim_path,
        tmp_path,
        config,
        rendering_modes=["composition", "custom"],
        sensor_modes=[],
    )
    assert not task._check_prim_files_exist(
        prim_path,
        tmp_path,
        config,
        rendering_modes=[],
        sensor_modes=["normals"],
    )
    default_info = {"images": []}
    task._record_existing_files(default_info, prim_path, tmp_path, config, tmp_path)
    assert {image["render_mode"] for image in default_info["images"]} == {
        "prim_only",
        "prim_with_stage",
    }

    prim_info = {"images": []}
    task._record_existing_files(
        prim_info,
        prim_path,
        tmp_path,
        config,
        tmp_path,
        rendering_modes=["prim_only", "composition", "custom"],
        sensor_modes=["depth"],
    )
    assert {image["render_mode"] for image in prim_info["images"]} == {
        "prim_only",
        "composition",
        "custom",
        "depth",
    }
    assert all(image["skipped"] is True for image in prim_info["images"])


def test_extract_metadata_display_material_and_hierarchy() -> None:
    stage, prim = _stage_with_mesh()
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()

    world = stage.GetPrimAtPath("/World")
    world.SetCustomDataByKey("annotation", "root-note")
    world.CreateAttribute(
        "omni:hoops:metadata:partNumber", Sdf.ValueTypeNames.String
    ).Set("PN-1")
    prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
        Vt.Vec3fArray([Gf.Vec3f(0.1, 0.2, 0.3)])
    )
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    material = _bind_mdl_material(stage, prim)
    subset = UsdGeom.Subset.Define(stage, "/World/Cube/Subset")
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)

    metadata = task._extract_prim_metadata(prim)
    assert metadata["type"] == "Mesh"
    assert metadata["active"] is True
    assert metadata["material"] == "/World/Looks/Mat"
    assert metadata["custom_data"]["annotation"] == "root-note"
    assert metadata["hoops_metadata"]["partNumber"] == "PN-1"
    assert "extent" in metadata
    assert metadata["has_transform"] is True

    assert task._extract_display_color(prim, listener) == pytest.approx([0.1, 0.2, 0.3])
    assert task._extract_display_color(world, listener) is None

    bindings = task._extract_material_bindings(prim, stage, listener)
    assert bindings["resolved"] == "/World/Looks/Mat"
    assert bindings["mdl_path"] == "OmniPBR.mdl"
    assert bindings["mdl_sub_identifier"] == "OmniPBR"
    assert bindings["subassignments"]["subset:Subset"] == "/World/Looks/Mat"

    model_node = SimpleNamespace(
        parent_path="/World",
        children_paths=["/World/Cube/Subset"],
    )
    fake_model = SimpleNamespace(
        get_prim=lambda path: model_node,
        get_ancestors=lambda path, include_self=False: [SimpleNamespace(path="/World")],
        get_collections_containing_prim=lambda path: [
            SimpleNamespace(name="setA", prim_path="/World/Collection")
        ],
    )
    hierarchy = task._extract_hierarchy_info(prim, fake_model)
    assert hierarchy["parent_path"] == "/World"
    assert hierarchy["collections"] == [
        {"name": "setA", "prim_path": "/World/Collection"}
    ]

    fallback = task._extract_hierarchy_info(prim, None)
    assert fallback["parent_path"] == "/World"
    assert fallback["children_paths"] == ["/World/Cube/Subset"]
    root_fallback = task._extract_hierarchy_info(world, None)
    assert root_fallback["parent_path"] is None


def test_extract_metadata_display_and_material_defensive_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()

    class FakeBoundable:
        def __init__(self, prim: object) -> None:
            self.prim = prim

        def GetExtentAttr(self):
            return SimpleNamespace(
                HasValue=lambda: True,
                Get=lambda: (_ for _ in ()).throw(RuntimeError("bad extent")),
            )

    class RaisingBindingAPI:
        def __init__(self, prim: object) -> None:
            self.prim = prim

        def GetDirectBinding(self):
            raise RuntimeError("bad binding")

    class FakeReferenceList:
        def GetAddedOrExplicitItems(self):
            return [SimpleNamespace(primPath=Sdf.Path("/Referenced"))]

    class FakeAttribute:
        def GetName(self) -> str:
            return "omni:hoops:metadata:broken"

        def Get(self):
            raise RuntimeError("bad attr")

    class FakePrim:
        def GetTypeName(self) -> str:
            return "Mesh"

        def GetPath(self) -> str:
            return "/World/Fake"

        def IsActive(self) -> bool:
            return True

        def IsA(self, schema: object) -> bool:
            return schema is FakeBoundable

        def HasAPI(self, api: object) -> bool:
            return api is RaisingBindingAPI

    fake_ancestor = SimpleNamespace(
        HasCustomData=lambda: False,
        GetPrimStack=lambda: [
            SimpleNamespace(referenceList=FakeReferenceList()),
            SimpleNamespace(referenceList=FakeReferenceList()),
        ],
        GetAttributes=lambda: [FakeAttribute()],
    )
    monkeypatch.setattr(prim_traversal.UsdGeom, "Boundable", FakeBoundable)
    monkeypatch.setattr(
        prim_traversal.UsdShade, "MaterialBindingAPI", RaisingBindingAPI
    )
    monkeypatch.setattr(task, "_traverse_to_root", lambda prim: [fake_ancestor])

    metadata = task._extract_prim_metadata(FakePrim())
    assert metadata["references"] == ["/Referenced"]
    assert "extent" not in metadata
    assert "material" not in metadata

    class ScalarColor:
        def __bool__(self) -> bool:
            return True

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> float:
            return 0.25

    class TruthyEmptyColor:
        def __bool__(self) -> bool:
            return True

        def __len__(self) -> int:
            return 0

    class FakeColorPrim:
        def __init__(self, value: object) -> None:
            self.value = value

        def HasAttribute(self, name: str) -> bool:
            return True

        def GetAttribute(self, name: str):
            return SimpleNamespace(HasValue=lambda: True, Get=lambda: self.value)

        def GetPath(self) -> str:
            return "/World/Color"

    assert task._extract_display_color(FakeColorPrim(ScalarColor()), listener) == [0.25]
    assert (
        task._extract_display_color(FakeColorPrim(TruthyEmptyColor()), listener) is None
    )


def test_extract_material_bindings_and_mdl_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()

    class FakeMaterial:
        def __init__(self, value: object = None) -> None:
            self.value = value

        def GetPath(self) -> str:
            return "/World/Looks/Fake"

    class FakeRel:
        def GetTargets(self):
            return [Sdf.Path("/World/Looks/Fake")]

        def IsValid(self) -> bool:
            return True

        def GetPrim(self):
            return SimpleNamespace(GetPath=lambda: Sdf.Path("/World"))

    class FakeMaterialPrim:
        def IsA(self, schema: object) -> bool:
            return schema is FakeMaterial

    class FakeStage:
        def GetPrimAtPath(self, path: str):
            return FakeMaterialPrim()

    class FakePrim:
        def __init__(self, *, fallback: bool) -> None:
            self.fallback = fallback

        def HasAPI(self, api: object) -> bool:
            return True

        def GetStage(self):
            return FakeStage()

        def GetPath(self) -> str:
            return "/World/Mesh"

        def GetChildren(self):
            return []

    class FakeBindingAPI:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def ComputeBoundMaterial(self):
            if self.prim.fallback:
                raise RuntimeError("old provider")
            return FakeMaterial()

        def GetDirectBindingRel(self):
            return FakeRel()

    monkeypatch.setattr(prim_traversal.UsdShade, "Material", FakeMaterial)
    monkeypatch.setattr(prim_traversal.UsdShade, "MaterialBindingAPI", FakeBindingAPI)
    monkeypatch.setattr(task, "_extract_mdl_paths_from_material", lambda *_: {})

    direct = task._extract_material_bindings(
        FakePrim(fallback=False), object(), listener
    )
    fallback = task._extract_material_bindings(
        FakePrim(fallback=True), object(), listener
    )

    assert direct["resolved"] == "/World/Looks/Fake"
    assert fallback["resolved"] == "/World/Looks/Fake"
    assert fallback["bound_at"] == "/World"
    assert fallback["inherited"] is True
    monkeypatch.setattr(
        task,
        "_extract_mdl_paths_from_material",
        USDPrimTraversalAndRenderingTask._extract_mdl_paths_from_material.__get__(
            task,
            type(task),
        ),
    )

    class BadAsset:
        def GetAssetPath(self):
            raise RuntimeError("no method path")

        @property
        def path(self):
            raise RuntimeError("no property path")

        def __str__(self) -> str:
            return "asset-as-text"

    class FakeShaderPrim:
        def GetPath(self) -> str:
            return "/World/Looks/Fake/Shader"

        def GetAttribute(self, name: str):
            if name == "info:mdl:sourceAsset":
                return SimpleNamespace(IsValid=lambda: True, Get=lambda: BadAsset())
            return SimpleNamespace(IsValid=lambda: True, Get=lambda: "SubId")

    class FakeShader:
        def GetPrim(self):
            return FakeShaderPrim()

    class FakeShaderWrapper:
        def __init__(self, prim: FakeShaderPrim) -> None:
            self.prim = prim

        def GetPath(self) -> str:
            return self.prim.GetPath()

        def GetPrim(self):
            return self.prim

    class FakeSurfaceOutput:
        def HasConnectedSource(self) -> bool:
            return True

        def GetConnectedSource(self):
            return (FakeShader(),)

    class FakeMdlMaterial:
        def GetSurfaceOutput(self, name: str):
            return FakeSurfaceOutput()

        def GetPath(self) -> str:
            return "/World/Looks/Fake"

    monkeypatch.setattr(prim_traversal.UsdShade, "Shader", FakeShaderWrapper)
    mdl_info = task._extract_mdl_paths_from_material(FakeMdlMaterial(), listener)
    assert mdl_info["mdl_path"] == "asset-as-text"
    assert mdl_info["mdl_sub_identifier"] == "SubId"


def test_collect_prims_filters_and_type_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, prim = _stage_with_mesh()
    UsdGeom.Xform.Define(stage, "/World/Group")
    UsdGeom.Cube.Define(stage, "/World/NativeCube")
    UsdGeom.Sphere.Define(stage, "/World/NativeSphere")
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()

    assert task._get_prim_type_from_string("UsdGeom.Mesh", listener) is UsdGeom.Mesh
    assert task._get_prim_type_from_string("Mesh", listener) is UsdGeom.Mesh
    assert (
        task._get_prim_type_from_string("UsdShade.Material", listener)
        is UsdShade.Material
    )
    assert task._get_prim_type_from_string("UsdLux.DistantLight", listener) is not None
    assert task._get_prim_type_from_string("UsdSkel.Root", listener) is not None
    task._get_prim_type_from_string("UsdVol.NotARealSchema", listener)
    assert task._matches_type_name_fallback(prim, "Mesh") is False
    assert task._matches_type_name_fallback(prim, "UsdGeom.Mesh") is False
    volume_prim = stage.DefinePrim("/World/V", "Volume")
    assert task._matches_type_name_fallback(volume_prim, "UsdVol.Volume") is True
    assert task._get_prim_type_from_string("Unknown.Type", listener) is None
    assert task._matches_type_name_fallback(prim, "UsdGeom.Mesh") is False

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pxr" and "UsdVol" in fromlist:
            raise ImportError("UsdVol unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert task._get_prim_type_from_string("UsdVol.Volume", listener) is None

    def fake_usdvol_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pxr" and "UsdVol" in fromlist:
            return SimpleNamespace(UsdVol=SimpleNamespace(Volume=object))
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_usdvol_import)
    assert task._get_prim_type_from_string("UsdVol.NotARealSchema", listener) is None

    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setattr(
        prim_traversal.Tf,
        "Type",
        SimpleNamespace(FindByName=lambda type_name: SimpleNamespace(pythonClass=dict)),
    )
    assert task._get_prim_type_from_string("Plugin.Schema", listener) is dict

    collected = task._collect_prims(
        stage,
        {"types": ["UsdGeom.Mesh"], "exclude_paths": ["/World/Group"]},
        listener,
    )
    assert collected == ["/World/Cube"]

    expected_gprims = [
        "/World/Cube",
        "/World/NativeCube",
        "/World/NativeSphere",
    ]
    # USD builds differ on whether UsdVol.Volume derives from UsdGeom.Gprim.
    if volume_prim.IsA(UsdGeom.Gprim):
        expected_gprims.append("/World/V")
    assert (
        task._collect_prims(
            stage,
            {"types": ["UsdGeom.Gprim"], "skip_instances": False},
            listener,
        )
        == expected_gprims
    )

    assert task._collect_prims(stage, {"paths": ["/World/Cube"]}, listener) == [
        "/World/Cube"
    ]
    assert task._collect_prims(
        stage,
        {"paths": ["/World/Cube"], "skip_instances": True},
        listener,
    ) == ["/World/Cube"]
    assert task._collect_prims(
        stage,
        {"types": ["UsdGeom.Mesh"], "root_prim": "/World"},
        listener,
    ) == ["/World/Cube"]
    assert task._collect_prims(
        stage,
        {"types": ["UsdGeom.Mesh"], "root_prim": "/Missing"},
        listener,
    ) == ["/World/Cube"]

    monkeypatch.setattr(task, "_get_prim_type_from_string", lambda *_: None)
    monkeypatch.setattr(
        task, "_matches_type_name_fallback", lambda prim, type_name: True
    )
    assert "/World/Cube" in task._collect_prims(
        stage,
        {"types": ["UsdVol.Volume"]},
        listener,
    )


def test_collect_prims_excludes_non_evidence_gprims_before_stage_preparation() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Cube.Define(stage, "/World/DefaultCube")
    render_sphere = UsdGeom.Sphere.Define(stage, "/World/RenderSphere")
    render_sphere.CreatePurposeAttr(UsdGeom.Tokens.render)

    guide_parent = UsdGeom.Xform.Define(stage, "/World/GuideScope")
    guide_parent.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Sphere.Define(stage, "/World/GuideScope/GuideSphere")

    transparent_cube = UsdGeom.Cube.Define(stage, "/World/TransparentCube")
    transparent_cube.CreateDisplayOpacityAttr(Vt.FloatArray([0.0]))

    degenerate_cube = UsdGeom.Cube.Define(stage, "/World/DegenerateCube")
    degenerate_cube.CreateSizeAttr(0.0)

    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    diagnostics: list[dict[str, object]] = []
    filters = {
        "types": ["UsdGeom.Gprim"],
        "skip_instances": False,
        "allowed_purposes": ["default", "render"],
        "skip_fully_transparent": True,
        "skip_unusable_bbox": True,
    }

    selected = task._collect_prims(
        stage,
        filters,
        listener,
        diagnostics=diagnostics,
    )

    assert selected == ["/World/DefaultCube", "/World/RenderSphere"]
    assert diagnostics == [
        {
            "prim_path": "/World/GuideScope/GuideSphere",
            "reason": "purpose_not_allowed",
            "purpose": "guide",
            "allowed_purposes": ["default", "render"],
        },
        {
            "prim_path": "/World/TransparentCube",
            "reason": "fully_transparent",
            "max_display_opacity": 0.0,
        },
        {
            "prim_path": "/World/DegenerateCube",
            "reason": "unusable_bbox",
        },
    ]
    assert prim_traversal._summarize_prim_filter_diagnostics(diagnostics) == {
        "skipped_prims": diagnostics,
        "reason_counts": {
            "fully_transparent": 1,
            "purpose_not_allowed": 1,
            "unusable_bbox": 1,
        },
    }
    assert {call.args[0] for call in listener.info.call_args_list} >= {
        "Skipped 1 render-ineligible prims (reason=fully_transparent)",
        "Skipped 1 render-ineligible prims (reason=purpose_not_allowed)",
        "Skipped 1 render-ineligible prims (reason=unusable_bbox)",
    }

    scalar_purpose_diagnostics: list[dict[str, object]] = []
    assert task._collect_prims(
        stage,
        {
            **filters,
            "allowed_purposes": "default",
        },
        listener,
        diagnostics=scalar_purpose_diagnostics,
    ) == ["/World/DefaultCube"]
    assert scalar_purpose_diagnostics[0] == {
        "prim_path": "/World/RenderSphere",
        "reason": "purpose_not_allowed",
        "purpose": "render",
        "allowed_purposes": ["default"],
    }

    specific_diagnostics: list[dict[str, object]] = []
    assert (
        task._collect_prims(
            stage,
            {
                **filters,
                "paths": ["/World/TransparentCube"],
            },
            listener,
            diagnostics=specific_diagnostics,
        )
        == []
    )
    assert specific_diagnostics == [
        {
            "prim_path": "/World/TransparentCube",
            "reason": "fully_transparent",
            "max_display_opacity": 0.0,
        }
    ]

    # The invalid guide/transparent/degenerate prims are removed before the
    # shared preparation call, so they cannot suppress the valid prim-only mode.
    config = RenderingConfig(
        camera_ordering=["+x+y+z"],
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
    )
    prepared = task._prepare_stages(
        stage,
        selected,
        ["prim_only"],
        config,
        {},
        listener,
    )
    assert prepared["prim_only"]["data"][2] == 2


def test_collect_prims_reuses_one_bbox_cache_for_the_effective_purposes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/DefaultCube")
    render_cube = UsdGeom.Cube.Define(stage, "/World/RenderCube")
    render_cube.CreatePurposeAttr(UsdGeom.Tokens.render)
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()

    real_bbox_cache = prim_traversal.UsdGeom.BBoxCache
    cache_calls: list[tuple[object, tuple[str, ...], bool]] = []

    def recording_bbox_cache(
        time_code: object,
        included_purposes: list[str],
        useExtentsHint: bool = False,
    ) -> object:
        cache_calls.append((time_code, tuple(included_purposes), useExtentsHint))
        return real_bbox_cache(
            time_code,
            included_purposes,
            useExtentsHint=useExtentsHint,
        )

    monkeypatch.setattr(prim_traversal.UsdGeom, "BBoxCache", recording_bbox_cache)

    selected = task._collect_prims(
        stage,
        {
            "types": ["UsdGeom.Gprim"],
            "skip_instances": False,
            "allowed_purposes": ["default", "render"],
            "skip_unusable_bbox": True,
        },
        listener,
    )

    assert selected == ["/World/DefaultCube", "/World/RenderCube"]
    assert len(cache_calls) == 1
    assert cache_calls[0][1:] == (("default", "render"), False)


def test_collect_prims_recovers_guide_only_rigid_body_assemblies() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")

    guide_only_body = UsdGeom.Xform.Define(stage, "/World/GuideOnlyBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(guide_only_body)
    for name in ("PanelA", "PanelB"):
        guide = UsdGeom.Cube.Define(
            stage,
            f"/World/GuideOnlyBody/{name}",
        )
        guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
        guide.CreateDisplayOpacityAttr(Vt.FloatArray([0.0]))
    rejected_guide = UsdGeom.Cube.Define(
        stage,
        "/World/GuideOnlyBody/RejectedGuide",
    )
    rejected_guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    rejected_guide.CreateSizeAttr(0.0)
    nested_body = UsdGeom.Xform.Define(
        stage,
        "/World/GuideOnlyBody/NestedBody",
    ).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(nested_body)
    nested_guide = UsdGeom.Cube.Define(
        stage,
        "/World/GuideOnlyBody/NestedBody/NestedPanel",
    )
    nested_guide.CreatePurposeAttr(UsdGeom.Tokens.guide)

    mixed_body = UsdGeom.Xform.Define(stage, "/World/MixedBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(mixed_body)
    UsdGeom.Cube.Define(stage, "/World/MixedBody/VisiblePanel")
    mixed_guide = UsdGeom.Cube.Define(stage, "/World/MixedBody/GuideHelper")
    mixed_guide.CreatePurposeAttr(UsdGeom.Tokens.guide)

    disabled_body = UsdGeom.Xform.Define(stage, "/World/DisabledBody").GetPrim()
    disabled_api = UsdPhysics.RigidBodyAPI.Apply(disabled_body)
    disabled_api.CreateRigidBodyEnabledAttr(False)
    disabled_guide = UsdGeom.Cube.Define(
        stage,
        "/World/DisabledBody/GuidePanel",
    )
    disabled_guide.CreatePurposeAttr(UsdGeom.Tokens.guide)

    static_guide = UsdGeom.Cube.Define(stage, "/World/StaticGuide")
    static_guide.CreatePurposeAttr(UsdGeom.Tokens.guide)

    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    diagnostics: list[dict[str, object]] = []
    assembly_target_members: dict[str, list[str]] = {}
    recovered_guide_gprim_targets: set[str] = set()
    filters = {
        "types": ["UsdGeom.Gprim"],
        "skip_instances": False,
        "allowed_purposes": ["default", "render"],
        "rigid_body_purpose_fallbacks": ["guide"],
        "skip_fully_transparent": True,
        "skip_unusable_bbox": True,
    }
    original_filters = deepcopy(filters)
    original_filter_bytes = json.dumps(filters, sort_keys=True).encode()
    selected = task._collect_prims(
        stage,
        filters,
        listener,
        diagnostics=diagnostics,
        assembly_target_members=assembly_target_members,
        recovered_guide_gprim_targets=recovered_guide_gprim_targets,
    )

    assert selected == [
        "/World/MixedBody/VisiblePanel",
        "/World/GuideOnlyBody",
        "/World/GuideOnlyBody/NestedBody",
    ]
    assert assembly_target_members == {
        "/World/GuideOnlyBody": [
            "/World/GuideOnlyBody/PanelA",
            "/World/GuideOnlyBody/PanelB",
        ],
        "/World/GuideOnlyBody/NestedBody": [
            "/World/GuideOnlyBody/NestedBody/NestedPanel",
        ],
    }
    assert recovered_guide_gprim_targets == set()
    assert {item["prim_path"] for item in diagnostics} == {
        "/World/GuideOnlyBody/PanelA",
        "/World/GuideOnlyBody/PanelB",
        "/World/GuideOnlyBody/RejectedGuide",
        "/World/GuideOnlyBody/NestedBody/NestedPanel",
        "/World/MixedBody/GuideHelper",
        "/World/DisabledBody/GuidePanel",
        "/World/StaticGuide",
    }
    assert all(item["reason"] == "purpose_not_allowed" for item in diagnostics)
    assert filters == original_filters
    assert json.dumps(filters, sort_keys=True).encode() == original_filter_bytes
    listener.info.assert_any_call(
        "Recovered guide-only rigid-body owner /World/GuideOnlyBody as one "
        "bounded assembly target from 2 purpose-filtered descendant(s)"
    )
    listener.info.assert_any_call(
        "Recovered guide-only rigid-body owner /World/GuideOnlyBody/NestedBody "
        "as one bounded assembly target from 1 purpose-filtered descendant(s)"
    )


def test_collect_prims_records_fallback_diagnostic_before_later_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    owner = UsdGeom.Xform.Define(stage, "/World/Owner")
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    guide = UsdGeom.Cube.Define(stage, "/World/Owner/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Cube.Define(stage, "/World/Later")
    diagnostics: list[dict[str, object]] = []
    filters = {
        "types": ["UsdGeom.Gprim"],
        "skip_instances": False,
        "allowed_purposes": ["default", "render"],
        "rigid_body_purpose_fallbacks": ["guide"],
        "skip_unusable_bbox": True,
    }
    original_diagnostic = prim_traversal._render_evidence_skip_diagnostic

    def raise_after_fallback_candidate(
        prim: Usd.Prim,
        current_filters: dict[str, object],
        bbox_cache: object | None = None,
    ) -> dict[str, object] | None:
        if str(prim.GetPath()) == "/World/Later":
            raise RuntimeError("later traversal failed")
        return original_diagnostic(prim, current_filters, bbox_cache)

    monkeypatch.setattr(
        prim_traversal,
        "_render_evidence_skip_diagnostic",
        raise_after_fallback_candidate,
    )

    with pytest.raises(RuntimeError, match="later traversal failed"):
        USDPrimTraversalAndRenderingTask()._collect_prims(
            stage,
            filters,
            Mock(),
            diagnostics=diagnostics,
            assembly_target_members={},
            recovered_guide_gprim_targets=set(),
        )

    assert diagnostics == [
        {
            "prim_path": "/World/Owner/Guide",
            "reason": "purpose_not_allowed",
            "purpose": "guide",
            "allowed_purposes": ["default", "render"],
        }
    ]


def test_collect_prims_recovers_guide_gprim_owner_as_one_leaf() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    normal_owner = UsdGeom.Cube.Define(stage, "/World/NormalBody")
    UsdPhysics.RigidBodyAPI.Apply(normal_owner.GetPrim())
    guide_owner = UsdGeom.Cube.Define(stage, "/World/GuideBody")
    guide_owner.CreatePurposeAttr(UsdGeom.Tokens.guide)
    guide_owner.CreateDisplayOpacityAttr(Vt.FloatArray([0.0]))
    UsdPhysics.RigidBodyAPI.Apply(guide_owner.GetPrim())
    nested = UsdGeom.Cube.Define(stage, "/World/GuideBody/NestedGuide")
    nested.CreatePurposeAttr(UsdGeom.Tokens.guide)
    nested.CreateDisplayOpacityAttr(Vt.FloatArray([0.0]))
    unrelated = UsdGeom.Cube.Define(stage, "/World/UnrelatedGuide")
    unrelated.CreatePurposeAttr(UsdGeom.Tokens.guide)
    source_text = stage.GetRootLayer().ExportToString()
    diagnostics: list[dict[str, object]] = []
    assembly_target_members: dict[str, list[str]] = {}
    recovered_guide_gprim_targets: set[str] = set()
    listener = Mock()

    selected = USDPrimTraversalAndRenderingTask()._collect_prims(
        stage,
        {
            "types": ["UsdGeom.Gprim"],
            "skip_instances": False,
            "allowed_purposes": ["default", "render"],
            "rigid_body_purpose_fallbacks": ["guide"],
            "skip_fully_transparent": True,
            "skip_unusable_bbox": True,
        },
        listener,
        diagnostics=diagnostics,
        assembly_target_members=assembly_target_members,
        recovered_guide_gprim_targets=recovered_guide_gprim_targets,
    )

    assert selected == ["/World/NormalBody", "/World/GuideBody"]
    assert len(selected) == len(set(selected))
    assert assembly_target_members == {}
    assert recovered_guide_gprim_targets == {"/World/GuideBody"}
    assert {item["prim_path"] for item in diagnostics} == {
        "/World/GuideBody",
        "/World/GuideBody/NestedGuide",
        "/World/UnrelatedGuide",
    }
    listener.info.assert_any_call(
        "Recovered guide-only rigid-body owner /World/GuideBody as one bounded "
        "leaf target from its accepted self candidate; 1 descendant guide "
        "candidate(s) remained diagnostic-only"
    )
    assert stage.GetRootLayer().ExportToString() == source_text


def test_collect_prims_does_not_recover_gprim_from_descendant_guide_only() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    owner = UsdGeom.Cube.Define(stage, "/World/TransparentBody")
    owner.CreateDisplayOpacityAttr(Vt.FloatArray([0.0]))
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    nested = UsdGeom.Cube.Define(stage, "/World/TransparentBody/NestedGuide")
    nested.CreatePurposeAttr(UsdGeom.Tokens.guide)
    source_text = stage.GetRootLayer().ExportToString()
    diagnostics: list[dict[str, object]] = []
    assembly_target_members: dict[str, list[str]] = {}
    recovered_guide_gprim_targets: set[str] = set()
    listener = Mock()

    selected = USDPrimTraversalAndRenderingTask()._collect_prims(
        stage,
        {
            "types": ["UsdGeom.Gprim"],
            "skip_instances": False,
            "allowed_purposes": ["default", "render"],
            "rigid_body_purpose_fallbacks": ["guide"],
            "skip_fully_transparent": True,
            "skip_unusable_bbox": True,
        },
        listener,
        diagnostics=diagnostics,
        assembly_target_members=assembly_target_members,
        recovered_guide_gprim_targets=recovered_guide_gprim_targets,
    )

    assert selected == []
    assert assembly_target_members == {}
    assert recovered_guide_gprim_targets == set()
    assert [(item["prim_path"], item["reason"]) for item in diagnostics] == [
        ("/World/TransparentBody", "fully_transparent"),
        ("/World/TransparentBody/NestedGuide", "purpose_not_allowed"),
    ]
    listener.info.assert_any_call(
        "Did not recover rigid-body Gprim owner /World/TransparentBody: only "
        "descendant guide candidates passed the render-evidence filters"
    )
    assert stage.GetRootLayer().ExportToString() == source_text


def test_collect_prims_does_not_recover_unsupported_rigid_owner() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    owner = UsdGeom.Scope.Define(stage, "/World/ScopeBody")
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    guide = UsdGeom.Cube.Define(stage, "/World/ScopeBody/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    source_text = stage.GetRootLayer().ExportToString()
    diagnostics: list[dict[str, object]] = []
    assembly_target_members: dict[str, list[str]] = {}
    recovered_guide_gprim_targets: set[str] = set()
    listener = Mock()

    selected = USDPrimTraversalAndRenderingTask()._collect_prims(
        stage,
        {
            "types": ["UsdGeom.Gprim"],
            "skip_instances": False,
            "allowed_purposes": ["default", "render"],
            "rigid_body_purpose_fallbacks": ["guide"],
            "skip_unusable_bbox": True,
        },
        listener,
        diagnostics=diagnostics,
        assembly_target_members=assembly_target_members,
        recovered_guide_gprim_targets=recovered_guide_gprim_targets,
    )

    assert selected == []
    assert assembly_target_members == {}
    assert recovered_guide_gprim_targets == set()
    assert [item["prim_path"] for item in diagnostics] == ["/World/ScopeBody/Guide"]
    listener.warning.assert_called_once_with(
        "Cannot recover guide-only rigid-body owner /World/ScopeBody: expected "
        "UsdGeom.Gprim or UsdGeom.Xform, found Scope"
    )
    assert stage.GetRootLayer().ExportToString() == source_text


def test_collect_prims_specific_render_target_suppresses_owner_fallback() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    owner = UsdGeom.Xform.Define(stage, "/World/Owner").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(owner)
    visible = UsdGeom.Cube.Define(stage, "/World/Owner/Visible")
    guide = UsdGeom.Cube.Define(stage, "/World/Owner/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    source_text = stage.GetRootLayer().ExportToString()
    filters = {
        "types": ["UsdGeom.Gprim"],
        "paths": [str(visible.GetPath()), str(guide.GetPath())],
        "allowed_purposes": ["default", "render"],
        "rigid_body_purpose_fallbacks": "guide",
        "skip_unusable_bbox": True,
    }
    diagnostics: list[dict[str, object]] = []
    assembly_target_members: dict[str, list[str]] = {}

    selected = USDPrimTraversalAndRenderingTask()._collect_prims(
        stage,
        filters,
        Mock(),
        diagnostics=diagnostics,
        assembly_target_members=assembly_target_members,
    )

    assert selected == ["/World/Owner/Visible"]
    assert assembly_target_members == {}
    assert diagnostics == [
        {
            "prim_path": "/World/Owner/Guide",
            "reason": "purpose_not_allowed",
            "purpose": "guide",
            "allowed_purposes": ["default", "render"],
        }
    ]
    assert filters["rigid_body_purpose_fallbacks"] == "guide"
    assert stage.GetRootLayer().ExportToString() == source_text


def test_collect_prims_does_not_duplicate_owner_already_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    owner = UsdGeom.Xform.Define(stage, "/World/Owner").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(owner)
    guide = UsdGeom.Cube.Define(stage, "/World/Owner/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    source_text = stage.GetRootLayer().ExportToString()
    owner_path = str(owner.GetPath())
    guide_path = str(guide.GetPath())
    owner_queries: list[str] = []

    def nearest_owner(prim: Usd.Prim) -> str | None:
        prim_path = str(prim.GetPath())
        owner_queries.append(prim_path)
        return owner_path if prim_path == guide_path else None

    monkeypatch.setattr(
        prim_traversal,
        "_nearest_enabled_rigid_body_owner",
        nearest_owner,
    )
    diagnostics: list[dict[str, object]] = []
    assembly_target_members: dict[str, list[str]] = {}

    selected = USDPrimTraversalAndRenderingTask()._collect_prims(
        stage,
        {
            "paths": [owner_path, guide_path],
            "allowed_purposes": ["default", "render"],
            "rigid_body_purpose_fallbacks": ["guide"],
            "skip_unusable_bbox": True,
        },
        Mock(),
        diagnostics=diagnostics,
        assembly_target_members=assembly_target_members,
    )

    assert selected == [owner_path]
    assert owner_queries == [owner_path, guide_path]
    assert assembly_target_members == {}
    assert [item["prim_path"] for item in diagnostics] == [guide_path]
    assert stage.GetRootLayer().ExportToString() == source_text


def test_bbox_purpose_contract_is_explicitly_propagated_from_prim_filters() -> None:
    task = USDPrimTraversalAndRenderingTask()
    config = RenderingConfig(bbox_purposes=("guide",))

    default_config = task._propagate_bbox_purposes({}, config)
    joint_config = task._propagate_bbox_purposes(
        {"allowed_purposes": ["default", "render"]},
        config,
    )

    assert default_config.bbox_purposes == ("default",)
    assert joint_config.bbox_purposes == ("default", "render")
    fallback_config = task._propagate_bbox_purposes(
        {
            "allowed_purposes": ["default", "render"],
            "rigid_body_purpose_fallbacks": ["guide"],
        },
        config,
    )
    assert fallback_config.bbox_purposes == ("default", "render")
    assert prim_traversal._bbox_purposes_from_filters(
        {"allowed_purposes": "render"}
    ) == ("render",)
    assert prim_traversal._allowed_purposes_from_filters({}) is None
    assert prim_traversal._allowed_purposes_from_filters(
        {"allowed_purposes": ["default", "render"]}
    ) == ("default", "render")
    fallback_filters = {"rigid_body_purpose_fallbacks": "guide"}
    assert prim_traversal._rigid_body_purpose_fallbacks_from_filters(
        fallback_filters
    ) == ("guide",)
    assert fallback_filters == {"rigid_body_purpose_fallbacks": "guide"}


def test_explicit_assembly_bbox_returns_none_without_members() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    source_text = stage.GetRootLayer().ExportToString()

    assert (
        prim_traversal._explicit_members_world_bbox(
            stage,
            (),
            ("default", "render"),
        )
        is None
    )
    assert stage.GetRootLayer().ExportToString() == source_text


def test_render_evidence_filter_helpers_cover_defensive_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    assert (
        prim_traversal._render_evidence_skip_diagnostic(
            xform,
            {"allowed_purposes": ["default"]},
        )
        is None
    )
    assert prim_traversal._has_usable_world_bbox(xform) is False
    cube = UsdGeom.Cube.Define(stage, "/World/Cube").GetPrim()
    assert (
        prim_traversal._render_evidence_skip_diagnostic(
            cube,
            {
                "allowed_purposes": "default",
                "skip_unusable_bbox": True,
            },
        )
        is None
    )

    class FakePrimvar:
        values: object = 0.5

        def __bool__(self) -> bool:
            return True

        def ComputeFlattened(self, _time_code: object) -> object:
            return self.values

    fake_primvar = FakePrimvar()

    class FakePrimvarsAPI:
        def __init__(self, _prim: object) -> None:
            pass

        def FindPrimvarWithInheritance(self, _name: str) -> FakePrimvar:
            return fake_primvar

    monkeypatch.setattr(prim_traversal.UsdGeom, "PrimvarsAPI", FakePrimvarsAPI)
    assert prim_traversal._effective_display_opacities(xform) == [0.5]

    class FallbackPrimvar(FakePrimvar):
        calls = 0

        def ComputeFlattened(self, _time_code: object) -> object:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("default-time failure")
            return [0.5]

    fake_primvar = FallbackPrimvar()
    with caplog.at_level(logging.DEBUG, logger=prim_traversal.__name__):
        assert prim_traversal._effective_display_opacities(xform) == [0.5]
    assert "displayOpacity fallback succeeded" in caplog.text

    fake_primvar = FakePrimvar()
    fake_primvar.values = ["invalid", float("nan"), 0.25]
    assert prim_traversal._effective_display_opacities(xform) == [0.25]
    assert "Ignored 1 malformed and 1 non-finite displayOpacity value" in caplog.text

    class ExplodingPrimvar(FakePrimvar):
        def ComputeFlattened(self, _time_code: object) -> object:
            raise RuntimeError("malformed primvar")

    exploding_primvar = ExplodingPrimvar()

    class ExplodingPrimvarsAPI(FakePrimvarsAPI):
        def FindPrimvarWithInheritance(self, _name: str) -> ExplodingPrimvar:
            return exploding_primvar

    monkeypatch.setattr(prim_traversal.UsdGeom, "PrimvarsAPI", ExplodingPrimvarsAPI)
    assert prim_traversal._effective_display_opacities(xform) == []
    assert "Could not flatten displayOpacity" in caplog.text

    class MissingPrimvarsAPI(FakePrimvarsAPI):
        def FindPrimvarWithInheritance(self, _name: str) -> None:
            return None

    monkeypatch.setattr(prim_traversal.UsdGeom, "PrimvarsAPI", MissingPrimvarsAPI)
    assert prim_traversal._effective_display_opacities(xform) == []


def test_require_matching_prims_omits_empty_filter_diagnostics() -> None:
    stage = Usd.Stage.CreateInMemory()
    task = USDPrimTraversalAndRenderingTask()

    with pytest.raises(RuntimeError) as exc_info:
        task._require_matching_prims(
            stage,
            [],
            {"types": ["UsdGeom.Mesh"]},
            {"skipped_prims": [], "reason_counts": {}},
        )

    assert "prim_filter_diagnostics" not in str(exc_info.value)


def test_collect_prims_skip_counters_with_fake_prims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()

    class FakePrim:
        def __init__(
            self,
            path: str,
            *,
            instance: bool = False,
            prototype: bool = False,
            invisible: bool = False,
            valid: bool = True,
        ) -> None:
            self.path = path
            self.instance = instance
            self.prototype = prototype
            self.invisible = invisible
            self.valid = valid

        def __bool__(self) -> bool:
            return self.valid

        def IsValid(self) -> bool:
            return self.valid

        def IsInstance(self) -> bool:
            return self.instance

        def IsInstanceProxy(self) -> bool:
            return False

        def IsInPrototype(self) -> bool:
            return self.prototype

        def IsA(self, prim_type: object) -> bool:
            return True

        def GetPath(self) -> str:
            return self.path

    class FakeStage:
        def __init__(self, prims: dict[str, FakePrim]) -> None:
            self.prims = prims

        def GetPrimAtPath(self, path: str) -> FakePrim:
            return self.prims.get(path, FakePrim(path, valid=False))

        def TraverseAll(self):
            return iter(self.prims.values())

    class FakeImageable:
        def __init__(self, prim: FakePrim) -> None:
            self.prim = prim

        def __bool__(self) -> bool:
            return True

        def ComputeVisibility(self):
            return (
                prim_traversal.UsdGeom.Tokens.invisible
                if self.prim.invisible
                else prim_traversal.UsdGeom.Tokens.inherited
            )

    monkeypatch.setattr(prim_traversal.UsdGeom, "Imageable", FakeImageable)

    specific_stage = FakeStage(
        {
            "/Inst": FakePrim("/Inst", instance=True),
            "/Proto": FakePrim("/Proto", prototype=True),
            "/Hidden": FakePrim("/Hidden", invisible=True),
        }
    )
    assert (
        task._collect_prims(
            specific_stage,
            {
                "paths": ["/Inst", "/Proto", "/Hidden"],
                "skip_instances": True,
                "skip_prototypes": True,
                "skip_invisible": True,
            },
            listener,
        )
        == []
    )

    traversal_stage = FakeStage(
        {
            "/Inst": FakePrim("/Inst", instance=True),
            "/Proto": FakePrim("/Proto", prototype=True),
            "/Hidden": FakePrim("/Hidden", invisible=True),
            "/Visible": FakePrim("/Visible"),
        }
    )
    monkeypatch.setattr(task, "_get_prim_type_from_string", lambda *_: object)
    assert task._collect_prims(
        traversal_stage,
        {
            "types": ["Any"],
            "root_prim": "/Missing",
            "skip_instances": True,
            "skip_prototypes": True,
            "skip_invisible": True,
        },
        listener,
    ) == ["/Visible"]


def test_propagate_root_and_collect_filter_prims_with_skips(tmp_path: Path) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    mesh_a = UsdGeom.Mesh.Define(stage, "/World/MeshA")
    mesh_b = UsdGeom.Mesh.Define(stage, "/World/MeshB")
    mesh_b.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(0, 1, 0),
            ]
        )
    )
    mesh_b.CreateFaceVertexCountsAttr([3])
    mesh_b.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh_b.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 0)]))
    mesh_b.GetPrim().CreateAttribute(
        "primvars:displayColor",
        Sdf.ValueTypeNames.Color3fArray,
    ).Set(Vt.Vec3fArray([Gf.Vec3f(0.4, 0.5, 0.6)]))
    _bind_mdl_material(stage, mesh_a.GetPrim())

    task = USDPrimTraversalAndRenderingTask()
    config = RenderingConfig(camera_ordering=["+x"], camera_name_prefix="Cam")
    propagated = task._propagate_root_prim(
        {"root_prim": "/World"},
        RenderingConfig(root_prim_path=None),
    )
    assert propagated.root_prim_path == "/World"
    assert (
        task._propagate_root_prim(
            {"root_prim": "/Other"},
            RenderingConfig(root_prim_path="/Already"),
        ).root_prim_path
        == "/Already"
    )

    for filename in ("MeshB_posx_prim_only.png", "MeshB_posx_depth.png"):
        path = prim_path_to_directory_structure("/World/MeshB", tmp_path, filename)
        Image.new("RGB", (2, 2), "red").save(path)

    object_store = Mock()
    object_store.get.side_effect = lambda key, default=None: {
        "usd_model": None,
        "meters_per_unit": 2.0,
        "stage_world_bbox": {
            "min": [0, 0, 0],
            "max": [2, 2, 2],
            "center": [1, 1, 1],
            "size": [2, 2, 2],
        },
    }.get(key, default)
    listener = Mock()

    prims_to_render, prim_data, total_images = task._collect_and_filter_prims(
        stage,
        ["/World/MeshA", "/World/MeshB"],
        {},
        config,
        ["prim_only"],
        {
            "extract_metadata": True,
            "extract_display_color": True,
            "extract_material_bindings": True,
            "extract_hierarchy": True,
            "skip_existing": True,
            "skip_existing_materials": True,
            "sensor_rendering_modes": ["depth"],
        },
        object_store,
        tmp_path,
        tmp_path,
        listener,
    )

    assert prims_to_render == []
    assert total_images == 2
    assert len(prim_data) == 1
    assert prim_data[0]["prim_path"] == "/World/MeshB"
    assert prim_data[0]["display_color"] == pytest.approx([0.4, 0.5, 0.6])
    assert prim_data[0]["world_bbox_meters"]["size"] == [2.0, 2.0, 0.0]
    assert {image["render_mode"] for image in prim_data[0]["images"]} == {
        "prim_only",
        "depth",
    }


def test_collect_and_filter_prims_bounds_only_explicit_assembly_members(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Owner")
    panel_a = UsdGeom.Cube.Define(stage, "/World/Owner/PanelA")
    panel_a.CreatePurposeAttr(UsdGeom.Tokens.guide)
    panel_a.GetSizeAttr().Set(20.0)
    panel_a.GetSizeAttr().Set(2.0, Usd.TimeCode(0))
    panel_b = UsdGeom.Cube.Define(stage, "/World/Owner/PanelB")
    panel_b.CreatePurposeAttr(UsdGeom.Tokens.guide)
    panel_b.GetSizeAttr().Set(20.0)
    panel_b.GetSizeAttr().Set(2.0, Usd.TimeCode(0))
    UsdGeom.Xformable(panel_b.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(4.0, 0.0, 0.0))
    rejected = UsdGeom.Cube.Define(stage, "/World/Owner/RejectedGuide")
    rejected.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Xformable(rejected.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(100.0, 0.0, 0.0)
    )

    config = RenderingConfig(
        bbox_purposes=("default", "render"),
        assembly_target_members={
            "/World/Owner": (
                "/World/Owner/PanelA",
                "/World/Owner/PanelB",
            )
        },
    )
    object_store = Mock()
    object_store.get.side_effect = lambda key, default=None: {
        "usd_model": None,
        "meters_per_unit": 0.5,
        "stage_world_bbox": None,
    }.get(key, default)

    prims_to_render, prim_data, total_images = (
        USDPrimTraversalAndRenderingTask()._collect_and_filter_prims(
            stage,
            ["/World/Owner"],
            {},
            config,
            ["prim_only"],
            {
                "extract_hierarchy": True,
                "skip_existing": False,
            },
            object_store,
            tmp_path,
            tmp_path,
            Mock(),
        )
    )

    assert prims_to_render == ["/World/Owner"]
    assert total_images == 0
    assert len(prim_data) == 1
    assert prim_data[0]["world_bbox"] == {
        "min": [-1.0, -1.0, -1.0],
        "max": [5.0, 1.0, 1.0],
        "center": [2.0, 0.0, 0.0],
        "size": [6.0, 2.0, 2.0],
    }
    assert prim_data[0]["world_bbox_meters"]["size"] == [3.0, 1.0, 1.0]


def test_collect_and_filter_prims_bounds_recovered_guide_gprim_as_leaf(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    owner = UsdGeom.Cube.Define(stage, "/World/GuideBody")
    owner.CreatePurposeAttr(UsdGeom.Tokens.guide)
    owner.GetSizeAttr().Set(20.0)
    owner.GetSizeAttr().Set(2.0, Usd.TimeCode(0))
    nested = UsdGeom.Cube.Define(stage, "/World/GuideBody/NestedGuide")
    nested.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Xformable(nested.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(100.0, 0.0, 0.0))
    config = RenderingConfig(
        bbox_purposes=("default", "render"),
        recovered_guide_gprim_targets=("/World/GuideBody",),
    )
    object_store = Mock()
    object_store.get.side_effect = lambda key, default=None: {
        "usd_model": None,
        "meters_per_unit": 0.5,
        "stage_world_bbox": None,
    }.get(key, default)

    prims_to_render, prim_data, total_images = (
        USDPrimTraversalAndRenderingTask()._collect_and_filter_prims(
            stage,
            ["/World/GuideBody"],
            {},
            config,
            ["prim_only"],
            {
                "extract_hierarchy": True,
                "skip_existing": False,
            },
            object_store,
            tmp_path,
            tmp_path,
            Mock(),
        )
    )

    assert prims_to_render == ["/World/GuideBody"]
    assert total_images == 0
    assert prim_data[0]["world_bbox"] == {
        "min": [-1.0, -1.0, -1.0],
        "max": [1.0, 1.0, 1.0],
        "center": [0.0, 0.0, 0.0],
        "size": [2.0, 2.0, 2.0],
    }
    assert prim_data[0]["world_bbox_meters"]["size"] == [1.0, 1.0, 1.0]


def test_collect_and_filter_prims_records_renderable_material_and_empty_color(
    tmp_path: Path,
) -> None:
    stage, prim = _stage_with_mesh()
    _bind_mdl_material(stage, prim)
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    object_store = Mock()
    object_store.get.side_effect = lambda key, default=None: {
        "usd_model": None,
        "meters_per_unit": None,
        "stage_world_bbox": None,
    }.get(key, default)

    prims_to_render, prim_data, total_images = task._collect_and_filter_prims(
        stage,
        ["/World/Cube"],
        {},
        RenderingConfig(camera_ordering=["+x"], camera_name_prefix="Cam"),
        ["prim_only"],
        {
            "extract_display_color": True,
            "extract_material_bindings": True,
            "extract_hierarchy": False,
            "skip_existing": False,
            "skip_existing_materials": False,
        },
        object_store,
        tmp_path,
        tmp_path,
        listener,
    )

    assert prims_to_render == ["/World/Cube"]
    assert total_images == 0
    assert prim_data[0]["material_bindings"]["resolved"] == "/World/Looks/Mat"
    assert "display_color" not in prim_data[0]
    assert any(
        "No display color found" in call.args[0]
        for call in listener.debug.call_args_list
    )


def test_prepare_upload_and_cleanup_stage_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    stage = object()
    listener = Mock()
    duplicate_calls: list[object] = []

    monkeypatch.setattr(
        prim_traversal,
        "prepare_prims_with_composition",
        lambda *args, **kwargs: (("highlight", ["HC"], 2), ("plain", ["PC"], 2)),
    )

    def fake_duplicate(input_stage: object) -> object:
        duplicate_calls.append(input_stage)
        return f"duplicate-{len(duplicate_calls)}"

    monkeypatch.setattr(prim_traversal, "duplicate_stage", fake_duplicate)
    monkeypatch.setattr(
        prim_traversal,
        "prepare_render_prims",
        lambda prepared_stage, prims, config, render_mode: (
            prepared_stage,
            ["Cam_posx"],
            len(prims),
        ),
    )

    config = RenderingConfig(
        per_mode_base_mode={"custom_original": "prim_only"},
        per_mode_use_original_materials={"custom_original": True},
    )
    prepared = task._prepare_stages(
        stage,
        ["/World/A"],
        ["composition", "prim_only", "prim_with_stage", "custom_original", "unknown"],
        config,
        {"sensor_rendering_modes": ["depth"]},
        listener,
    )

    assert prepared["composition"]["type"] == "composition"
    assert prepared["prim_only"]["type"] == "prim_only"
    assert prepared["prim_with_stage"]["type"] == "prim_with_stage"
    assert prepared["custom_original"]["config"].should_reset_materials is False
    assert "unknown" not in prepared

    monkeypatch.setattr(
        prim_traversal,
        "prepare_prims_with_composition",
        Mock(side_effect=RuntimeError("prepare failed")),
    )
    assert (
        task._prepare_stages(
            stage,
            ["/World/A"],
            ["composition"],
            config,
            {},
            listener,
        )
        == {}
    )

    assert task._upload_stages_to_s3(prepared, object(), {}, listener) == []

    upload_calls: list[object] = []

    def fake_export_stage_to_s3(
        stage_arg: object, **kwargs: object
    ) -> tuple[str, str | None]:
        upload_calls.append(stage_arg)
        if stage_arg == "plain":
            raise RuntimeError("plain upload failed")
        s3_uri = f"s3://bucket/{stage_arg}" if stage_arg != "duplicate-2" else None
        return f"https://example.invalid/{stage_arg}", s3_uri

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
        fake_export_stage_to_s3,
    )
    backend = RemoteRenderingBackend(api_key="key", use_data_uri=False)
    cleanup = task._upload_stages_to_s3(
        {
            "composition": prepared["composition"],
            "prim_only": prepared["prim_only"],
            "prim_with_stage": prepared["prim_with_stage"],
        },
        backend,
        {"usd_path": str(tmp_path / "scene.usda")},
        listener,
    )

    assert ("s3://bucket/highlight", backend.s3_profile) in cleanup
    assert any(call == "duplicate-1" for call in upload_calls)
    assert prepared["composition"]["highlight_url"].endswith("/highlight")
    assert "plain_url" not in prepared["composition"]

    def fake_success_export_stage_to_s3(
        stage_arg: object, **kwargs: object
    ) -> tuple[str, str]:
        return f"https://example.invalid/{stage_arg}", f"s3://bucket/{stage_arg}"

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
        fake_success_export_stage_to_s3,
    )
    success_prepared = {
        "composition": {
            "type": "composition",
            "data": (("highlight-success", ["HC"], 1), ("plain-success", ["PC"], 1)),
        }
    }
    success_cleanup = task._upload_stages_to_s3(
        success_prepared,
        backend,
        {"usd_path": str(tmp_path / "scene.usda")},
        listener,
    )
    assert success_prepared["composition"]["plain_url"].endswith("/plain-success")
    assert success_cleanup == [
        ("s3://bucket/highlight-success", backend.s3_profile),
        ("s3://bucket/plain-success", backend.s3_profile),
    ]

    deleted: list[tuple[str, str | None]] = []

    def fake_delete_s3_path(s3_uri: str, profile_name: str | None = None) -> None:
        deleted.append((s3_uri, profile_name))
        if "fail" in s3_uri:
            raise RuntimeError("delete failed")

    monkeypatch.setattr(prim_traversal, "delete_s3_path", fake_delete_s3_path)
    task._cleanup_s3(
        [("s3://bucket/ok", "profile"), ("s3://bucket/fail", None)],
        listener,
    )
    task._cleanup_s3([], listener)

    assert deleted == [("s3://bucket/ok", "profile"), ("s3://bucket/fail", None)]


def test_process_batch_saves_images_sensors_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    listener.event.side_effect = RuntimeError("event failed")
    backend = RemoteRenderingBackend(api_key="key")
    config = RenderingConfig(image_width=2, camera_name_prefix="Cam")
    prim_data = {"/World/A": {"images": []}, "/World/B": {"images": []}}

    captured: dict[str, object] = {}

    def fake_render_from_prepared_prims(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        captured["prepared_stage"] = args[1]
        captured["stage_url"] = kwargs["stage_url"]
        return {
            "results": [
                {
                    "camera": "Cam_posx",
                    "status": "success",
                    "blank_render_frames": [
                        {"frame": 0, "stats": {"blank": True, "reason": "solid"}}
                    ],
                    "prim_to_images": {
                        "/World/A": Image.new("RGB", (2, 2), "blue"),
                        "/World/B": None,
                        "/World/Missing": Image.new("RGB", (2, 2), "green"),
                    },
                    "prim_occlusion": {"/World/B": True},
                    "sensors": {
                        "linear_depth": {
                            0: np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float32)
                        },
                        "instance_id_segmentation": {
                            1: np.array([[0, 1], [2, 2]], dtype=np.uint32)
                        },
                        "ignored": {8: np.array([1], dtype=np.uint8)},
                    },
                }
            ]
        }

    monkeypatch.setattr(
        prim_traversal,
        "render_from_prepared_prims",
        fake_render_from_prepared_prims,
    )

    total_images, failures, prim_images = task._process_batch(
        0,
        2,
        ["/World/A", "/World/B"],
        prim_data,
        {
            "prim_only": {
                "type": "prim_only",
                "data": ("template-stage", ["Cam_posx"], 2),
                "config": config,
                "stage_url": "https://example.invalid/stage.usd",
            }
        },
        backend,
        "prim_only",
        tmp_path,
        tmp_path,
        1,
        2,
        listener,
        sensor_modes=["linear_depth", "instance_id_segmentation"],
        image_height=2,
    )

    assert captured == {
        "prepared_stage": None,
        "stage_url": "https://example.invalid/stage.usd",
    }
    assert failures == []
    assert total_images == 3
    assert {image["render_mode"] for image in prim_images["/World/A"]} == {
        "prim_only",
        "linear_depth",
    }
    assert prim_images["/World/A"][0]["blank_render"] is True
    assert prim_images["/World/B"][0]["render_mode"] == "instance_id_segmentation"

    missing_total, missing_failures, missing_images = task._process_batch(
        0,
        1,
        ["/World/A"],
        prim_data,
        {},
        backend,
        "missing",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )
    assert (missing_total, missing_failures, missing_images) == (0, [], {})

    monkeypatch.setattr(
        prim_traversal,
        "render_from_prepared_prims",
        Mock(side_effect=RuntimeError("render failed")),
    )
    _, failed_batches, _ = task._process_batch(
        0,
        1,
        ["/World/A"],
        prim_data,
        {
            "prim_only": {
                "type": "prim_only",
                "data": ("template-stage", ["Cam_posx"], 1),
                "config": config,
                "stage_url": "https://example.invalid/stage.usd",
            }
        },
        backend,
        "prim_only",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )
    assert failed_batches[0]["error"] == "render failed"


def test_process_batch_composition_uses_urls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    backend = RemoteRenderingBackend(api_key="key")
    captured: dict[str, object] = {}

    def fake_render_from_prepared_composition(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        captured["highlight_stage"] = args[1]
        captured["plain_stage"] = args[3]
        captured["highlight_url"] = kwargs["highlight_url"]
        captured["plain_url"] = kwargs["plain_url"]
        return {"results": []}

    monkeypatch.setattr(
        prim_traversal,
        "render_from_prepared_composition",
        fake_render_from_prepared_composition,
    )

    total_images, failures, prim_images = task._process_batch(
        1,
        2,
        ["/World/A", "/World/B"],
        {"/World/B": {"images": []}},
        {
            "composition": {
                "type": "composition",
                "data": (("highlight", ["HC"], 2), ("plain", ["PC"], 2)),
                "config": RenderingConfig(image_width=2),
                "highlight_url": "https://example.invalid/highlight.usd",
                "plain_url": "https://example.invalid/plain.usd",
            }
        },
        backend,
        "composition",
        tmp_path,
        tmp_path,
        1,
        1,
        Mock(),
        sensor_modes=["depth"],
    )

    assert (total_images, failures, prim_images) == (0, [], {})
    assert captured == {
        "highlight_stage": None,
        "plain_stage": None,
        "highlight_url": "https://example.invalid/highlight.usd",
        "plain_url": "https://example.invalid/plain.usd",
    }


def test_process_batch_sync_duplicates_stages_and_saves_mode_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    backend = RemoteRenderingBackend(api_key="key")
    listener = Mock()
    duplicate_calls: list[object] = []

    def fake_duplicate_stage(stage: object) -> str:
        duplicate_calls.append(stage)
        return f"duplicate-{stage}"

    monkeypatch.setattr(
        "world_understanding.utils.usd.stage.duplicate_stage",
        fake_duplicate_stage,
    )

    def fake_render_from_prepared_composition(*args: object, **kwargs: object):
        return {
            "results": [
                {
                    "camera": "Front",
                    "status": "success",
                    "prim_to_images": {
                        "/World/A": Image.new("RGB", (2, 2), "red"),
                    },
                    "sensors": {
                        "depth": {
                            0: np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float32),
                            1: np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                        },
                        "bad": {0: np.array([1, 2, 3, 4, 5], dtype=np.uint8)},
                    },
                },
                {
                    "camera": "NoSensors",
                    "status": "success",
                    "prim_to_images": {},
                },
            ]
        }

    monkeypatch.setattr(
        prim_traversal,
        "render_from_prepared_composition",
        fake_render_from_prepared_composition,
    )

    total_images, failures, prim_images = task._process_batch(
        0,
        2,
        ["/World/A", "/World/Missing"],
        {"/World/A": {"images": []}},
        {
            "composition": {
                "type": "composition",
                "data": (("highlight", ["Front"], 1), ("plain", ["Front"], 1)),
                "config": RenderingConfig(image_width=2),
            }
        },
        backend,
        "composition",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
        sensor_modes=["depth"],
        image_height=2,
    )

    assert duplicate_calls[:2] == ["highlight", "plain"]
    assert failures == []
    assert total_images == 2
    assert {image["render_mode"] for image in prim_images["/World/A"]} == {
        "composition",
        "depth",
    }

    def fake_render_from_prepared_prims(*args: object, **kwargs: object):
        return {
            "results": [
                {
                    "camera": "Front",
                    "status": "success",
                    "prim_to_images": {
                        "/World/A": Image.new("RGB", (2, 2), "blue"),
                    },
                }
            ]
        }

    monkeypatch.setattr(
        prim_traversal,
        "render_from_prepared_prims",
        fake_render_from_prepared_prims,
    )

    prim_with_stage = task._process_batch(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "prim_with_stage": {
                "type": "prim_with_stage",
                "data": ("stage", ["Front"], 1),
                "config": RenderingConfig(image_width=2),
            }
        },
        backend,
        "prim_with_stage",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
        sensor_modes=["depth"],
    )
    assert prim_with_stage[0] == 1
    assert prim_with_stage[2]["/World/A"][0]["path"].endswith("_prim_with_stage.png")

    custom = task._process_batch(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "custom": {
                "type": "prim_only",
                "data": ("stage", ["Front"], 1),
                "config": RenderingConfig(image_width=2),
            }
        },
        backend,
        "custom",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )
    assert custom[2]["/World/A"][0]["path"].endswith("_custom.png")


def test_run_success_sequential_updates_context_and_handles_event_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage, _ = _stage_with_mesh()
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    listener.event.side_effect = RuntimeError("event failed")
    object_store = Mock()
    config = RenderingConfig(camera_ordering=["+x"])
    object_store.get.side_effect = lambda key, default=None: {
        "usd_stage": stage,
        "rendering_backend": object(),
        "rendering_config": config,
        "usd_model": None,
    }.get(key, default)

    monkeypatch.setattr(
        task, "_collect_prims", Mock(return_value=["/World/A", "/World/B"])
    )
    monkeypatch.setattr(
        task,
        "_collect_and_filter_prims",
        Mock(
            return_value=(
                ["/World/A", "/World/B"],
                [
                    {"prim_path": "/World/A", "images": [], "metadata": {}},
                    {"prim_path": "/World/B", "images": [], "metadata": {}},
                ],
                0,
            )
        ),
    )
    monkeypatch.setattr(
        task,
        "_prepare_stages",
        Mock(return_value={"prim_only": {"type": "prim_only"}}),
    )
    monkeypatch.setattr(
        task,
        "_upload_stages_to_s3",
        Mock(return_value=[("s3://bucket/temp", "profile")]),
    )
    cleanup = Mock()
    monkeypatch.setattr(task, "_cleanup_s3", cleanup)
    monkeypatch.setattr(task, "_check_blank_dataset_renders", Mock())

    def fake_process_batch(
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        *args: object,
        **kwargs: object,
    ) -> tuple[int, list[dict[str, object]], dict[str, list[dict[str, object]]]]:
        prim_path = prims_to_render[batch_start]
        return (
            1,
            [{"batch_start": batch_start, "render_mode": "prim_only"}],
            {prim_path: [{"path": f"{prim_path.strip('/')}.png"}]},
        )

    monkeypatch.setattr(task, "_process_batch_with_retry_split", fake_process_batch)

    context = {
        "event_listener": listener,
        "prim_filters": {"types": ["UsdGeom.Mesh"]},
        "render_output_dir": str(tmp_path / "renders"),
        "batch_size": 1,
        "rendering_modes": ["prim_only"],
        "sensor_rendering_modes": ["depth"],
    }

    result = task.run(context, object_store)

    assert result["rendered_prims"] == ["/World/A", "/World/B"]
    assert result["total_images_rendered"] == 2
    assert len(result["failed_batches"]) == 2
    assert all(item["images"] for item in result["prim_data"])
    cleanup.assert_called_once_with([("s3://bucket/temp", "profile")], listener)
    object_store.set.assert_any_call("prim_data", result["prim_data"])


def test_run_parallel_records_batch_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage, _ = _stage_with_mesh()
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    object_store = Mock()
    object_store.get.side_effect = lambda key, default=None: {
        "usd_stage": stage,
        "rendering_backend": object(),
        "rendering_config": RenderingConfig(camera_ordering=["+x"]),
        "usd_model": None,
    }.get(key, default)

    def collect_with_recovered_guide(
        *args: object,
        **kwargs: object,
    ) -> list[str]:
        recovered = kwargs["recovered_guide_gprim_targets"]
        assert isinstance(recovered, set)
        recovered.add("/World/A")
        return ["/World/A"]

    monkeypatch.setattr(task, "_collect_prims", collect_with_recovered_guide)
    collect_and_filter = Mock(
        return_value=(
            ["/World/A"],
            [{"prim_path": "/World/A", "images": [], "metadata": {}}],
            0,
        )
    )
    monkeypatch.setattr(task, "_collect_and_filter_prims", collect_and_filter)
    monkeypatch.setattr(task, "_prepare_stages", Mock(return_value={}))
    monkeypatch.setattr(task, "_upload_stages_to_s3", Mock(return_value=[]))
    monkeypatch.setattr(task, "_cleanup_s3", Mock())
    monkeypatch.setattr(task, "_check_blank_dataset_renders", Mock())

    def fake_process_batch(
        batch_start: int,
        batch_end: int,
        prims_to_render: list[str],
        prim_data: dict[str, dict[str, object]],
        prepared_stages: dict[str, dict[str, object]],
        rendering_backend: object,
        render_mode: str,
        *args: object,
        **kwargs: object,
    ) -> tuple[int, list[dict[str, object]], dict[str, list[dict[str, object]]]]:
        if render_mode == "composition":
            raise RuntimeError("parallel render failed")
        return (
            1,
            [{"batch_start": batch_start, "render_mode": render_mode}],
            {"/World/A": [{"path": "World/A.png"}]},
        )

    monkeypatch.setattr(task, "_process_batch_with_retry_split", fake_process_batch)

    context = {
        "event_listener": listener,
        "render_output_dir": str(tmp_path / "renders"),
        "output_dir": str(tmp_path),
        "batch_size": 1,
        "num_workers": 2,
        "rendering_modes": ["prim_only", "composition"],
        "sensor_rendering_modes": ["depth"],
    }

    result = task.run(context, object_store)

    assert result["total_images_rendered"] == 1
    assert result["prim_data"][0]["images"] == [{"path": "World/A.png"}]
    assert any(
        item.get("render_mode") == "prim_only" for item in result["failed_batches"]
    )
    assert any(
        item.get("render_mode") == "composition"
        and "parallel render failed" in item.get("error", "")
        for item in result["failed_batches"]
    )
    propagated_config = collect_and_filter.call_args.args[3]
    assert propagated_config.recovered_guide_gprim_targets == ("/World/A",)
    assert context["recovered_guide_gprim_targets"] == ["/World/A"]
    object_store.set.assert_any_call(
        "stage_up_axis", str(UsdGeom.GetStageUpAxis(stage))
    )


def test_run_records_missing_prims_without_failing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage, _ = _stage_with_mesh()
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    object_store = Mock()
    object_store.get.side_effect = lambda key, default=None: {
        "usd_stage": stage,
        "rendering_backend": object(),
        "rendering_config": RenderingConfig(camera_ordering=["+x"]),
        "usd_model": None,
    }.get(key, default)

    monkeypatch.setattr(
        task, "_collect_prims", Mock(return_value=["/World/A", "/World/B"])
    )
    monkeypatch.setattr(
        task,
        "_collect_and_filter_prims",
        Mock(
            return_value=(
                ["/World/A", "/World/B"],
                [
                    {"prim_path": "/World/A", "images": [], "metadata": {}},
                    {"prim_path": "/World/B", "images": [], "metadata": {}},
                ],
                0,
            )
        ),
    )
    monkeypatch.setattr(task, "_prepare_stages", Mock(return_value={"prim_only": {}}))
    monkeypatch.setattr(task, "_upload_stages_to_s3", Mock(return_value=[]))
    monkeypatch.setattr(task, "_cleanup_s3", Mock())
    monkeypatch.setattr(task, "_check_blank_dataset_renders", Mock())
    monkeypatch.setattr(
        task,
        "_process_batch_with_retry_split",
        Mock(return_value=(1, [], {"/World/A": [{"path": "World/A.png"}]})),
    )

    result = task.run(
        {
            "event_listener": listener,
            "render_output_dir": str(tmp_path / "renders"),
            "output_dir": str(tmp_path),
            "batch_size": 2,
            "rendering_modes": ["prim_only"],
            "fail_on_missing_prim_images": False,
        },
        object_store,
    )

    assert result["missing_image_prims"] == ["/World/B"]
    listener.warning.assert_called_once()


def test_blank_render_helper_matrix(tmp_path: Path) -> None:
    task = USDPrimTraversalAndRenderingTask()
    context = {
        "failed_batches": [
            "not-a-dict",
            {"render_mode": "composition"},
            {"blank_render": True, "render_mode": "depth", "prim_path": "/Sensor"},
            {"blank_render": True, "render_mode": "other", "prim_path": "/Other"},
            {
                "blank_render": True,
                "render_mode": "composition",
                "prim_path": "/World/A",
                "path": "a.png",
                "view": "front",
                "camera": "Cam_front",
                "error": "blank",
            },
        ]
    }

    failures = task._blank_render_failure_candidates(
        context,
        rgb_modes=["composition"],
        sensor_modes=["depth"],
    )
    assert failures == [
        {
            "prim_path": "/World/A",
            "path": "a.png",
            "render_mode": "composition",
            "view": "front",
            "camera": "Cam_front",
            "stats": {"blank": True, "reason": "remote_blank_render"},
            "error": "blank",
            "blank_render": True,
        }
    ]
    assert task._dedupe_blank_render_failures(failures + failures) == failures
    assert (
        task._dedupe_blank_render_failures(
            failures,
            existing_keys={("/World/A", "composition")},
        )
        == []
    )
    assert task._blank_render_candidate_key(
        {"camera": "Cam", "frame": 3, "path": "p.png", "view": "v"}
    ) == ("unknown:Cam:3:p.png:v", "")

    frame_failures = task._blank_render_failures_from_results(
        {
            "results": [
                {
                    "camera": "Cam",
                    "status": "blank_render",
                    "blank_render_frames": [
                        {"frame": "bad"},
                        {"frame": 99},
                        {"frame": 2},
                    ],
                    "error": "blank",
                }
            ]
        },
        batch_start=2,
        batch_prims=["/World/B"],
        render_mode="composition",
    )
    assert [failure["prim_path"] for failure in frame_failures] == [
        None,
        None,
        "/World/B",
    ]
    all_frame_failures = task._blank_render_failures_from_results(
        {
            "results": [
                {
                    "camera": "Cam",
                    "status": "blank_render",
                    "error": "all blank",
                }
            ]
        },
        batch_start=5,
        batch_prims=["/World/C", "/World/D"],
        render_mode="composition",
    )
    assert [failure["prim_path"] for failure in all_frame_failures] == [
        "/World/C",
        "/World/D",
    ]

    assert (
        task._blank_render_frame_stats_by_prim(
            {"blank_render_frames": "bad"},
            batch_start=0,
            batch_prims=["/World/A"],
        )
        == {}
    )
    assert task._blank_render_frame_stats_by_prim(
        {
            "blank_render_frames": [
                "bad",
                {"frame": "bad"},
                {"frame": 99},
                {"frame": 0},
            ]
        },
        batch_start=0,
        batch_prims=["/World/A"],
    ) == {"/World/A": {"blank": True, "reason": "remote_blank_render"}}

    candidates = task._dataset_render_candidates(
        [
            {
                "prim_path": "/World/A",
                "images": [
                    "not-a-dict",
                    {"render_mode": "composition"},
                    {"path": "depth.png", "render_mode": "depth"},
                    {"path": "other.png", "render_mode": "other"},
                    {
                        "path": "a_prim_only.png",
                        "render_mode": "prim_only",
                        "view": "front",
                        "camera": "Cam_front",
                    },
                    {
                        "path": "a_composition.png",
                        "render_mode": "composition",
                        "blank_render": True,
                        "stats": {"blank": True, "reason": "solid"},
                    },
                ],
            }
        ],
        tmp_path,
        rgb_modes=["prim_only", "composition"],
        sensor_modes=["depth"],
    )
    assert len(candidates) == 1
    assert candidates[0]["render_mode"] == "composition"
    assert candidates[0]["blank_render"] is True
    assert candidates[0]["stats"]["reason"] == "solid"


def test_blank_dataset_candidates_cover_analysis_and_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    for filename in ("already.png", "clear.png", "error.png", "dark.png"):
        Image.new("RGB", (2, 2), "black").save(tmp_path / filename)

    def fake_analyze_image_blankness(path: Path):
        name = Path(path).name
        if name == "error.png":
            raise RuntimeError("cannot inspect")
        is_blank = name == "dark.png"
        return SimpleNamespace(
            blank=is_blank,
            to_dict=lambda: {"blank": is_blank, "reason": name},
        )

    monkeypatch.setattr(
        prim_traversal,
        "analyze_image_blankness",
        fake_analyze_image_blankness,
    )

    context = {
        "failed_batches": [
            {
                "blank_render": True,
                "render_mode": "composition",
                "prim_path": "/World/A",
                "path": "already.png",
            },
            {
                "blank_render": True,
                "render_mode": "composition",
                "prim_path": "/World/B",
                "path": "clear.png",
            },
            {
                "blank_render": True,
                "render_mode": "composition",
                "prim_path": "/World/Z",
                "path": "remote.png",
            },
        ],
        "fail_on_blank_dataset_renders": False,
    }
    prim_data = [
        {
            "prim_path": "/World/A",
            "images": [
                {
                    "path": "already.png",
                    "render_mode": "composition",
                    "blank_render": True,
                    "stats": {"blank": True, "reason": "remote"},
                }
            ],
        },
        {
            "prim_path": "/World/B",
            "images": [{"path": "clear.png", "render_mode": "composition"}],
        },
        {
            "prim_path": "/World/C",
            "images": [{"path": "error.png", "render_mode": "composition"}],
        },
        {
            "prim_path": "/World/D",
            "images": [{"path": "dark.png", "render_mode": "composition"}],
        },
    ]

    task._check_blank_dataset_renders(
        prim_data,
        tmp_path,
        rgb_modes=["composition"],
        sensor_modes=[],
        listener=listener,
        context=context,
    )

    blank_prim_paths = {item["prim_path"] for item in context["blank_renders"]}
    assert blank_prim_paths == {
        "/World/A",
        "/World/B",
        "/World/C",
        "/World/D",
        "/World/Z",
    }
    assert context["blank_render_checked_count"] == 5
    listener.warning.assert_called()


def test_blank_dataset_candidates_return_when_all_nonblank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    image_path = tmp_path / "clear.png"
    Image.new("RGB", (2, 2), "white").save(image_path)
    monkeypatch.setattr(
        prim_traversal,
        "analyze_image_blankness",
        lambda path: SimpleNamespace(blank=False, to_dict=lambda: {"blank": False}),
    )
    context: dict[str, object] = {}

    task._check_blank_dataset_renders(
        [
            {
                "prim_path": "/World/A",
                "images": [{"path": str(image_path), "render_mode": "composition"}],
            }
        ],
        tmp_path,
        rgb_modes=["composition"],
        sensor_modes=[],
        listener=Mock(),
        context=context,
    )

    assert context == {}


def test_blank_dataset_guard_no_candidate_and_warning_paths(tmp_path: Path) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    context: dict[str, object] = {}
    task._check_blank_dataset_renders(
        [],
        tmp_path,
        rgb_modes=["composition"],
        sensor_modes=[],
        listener=listener,
        context=context,
    )
    assert context == {}

    context = {
        "failed_batches": [
            {
                "blank_render": True,
                "render_mode": "composition",
                "prim_path": "/World/A",
            }
        ],
        "blank_render_failure_threshold": 2.0,
    }
    task._check_blank_dataset_renders(
        [],
        tmp_path,
        rgb_modes=["composition"],
        sensor_modes=[],
        listener=listener,
        context=context,
    )

    assert context["blank_render_checked_count"] == 1
    assert context["blank_renders"][0]["prim_path"] == "/World/A"
    listener.warning.assert_called()


@pytest.mark.asyncio
async def test_arun_success_remote_records_missing_prims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage, _ = _stage_with_mesh()
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    object_store = Mock()
    backend = RemoteRenderingBackend(api_key="key")
    object_store.get.side_effect = lambda key, default=None: {
        "usd_stage": stage,
        "rendering_backend": backend,
        "rendering_config": RenderingConfig(camera_ordering=["+x"]),
        "usd_model": None,
    }.get(key, default)

    prims = ["/World/A", "/World/B", "/World/C", "/World/D"]
    monkeypatch.setattr(task, "_collect_prims", Mock(return_value=prims))
    monkeypatch.setattr(
        task,
        "_collect_and_filter_prims",
        Mock(
            return_value=(
                prims,
                [{"prim_path": prim, "images": [], "metadata": {}} for prim in prims],
                0,
            )
        ),
    )
    monkeypatch.setattr(task, "_prepare_stages", Mock(return_value={"prim_only": {}}))

    async def fake_upload_stages_to_s3_async(*args: object, **kwargs: object):
        return []

    monkeypatch.setattr(
        task, "_upload_stages_to_s3_async", fake_upload_stages_to_s3_async
    )
    monkeypatch.setattr(task, "_cleanup_s3", Mock())
    monkeypatch.setattr(task, "_check_blank_dataset_renders", Mock())
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.get_global_remote_render_limit",
        lambda: 3,
    )

    async def fake_process_batch_async_with_retry_split(
        *args: object, **kwargs: object
    ):
        return (
            0,
            len(prims),
            "prim_only",
            1,
            [{"batch_start": 0, "render_mode": "prim_only"}],
            {"/World/A": [{"path": "World/A.png"}]},
        )

    monkeypatch.setattr(
        task,
        "_process_batch_async_with_retry_split",
        fake_process_batch_async_with_retry_split,
    )

    result = await task.arun(
        {
            "event_listener": listener,
            "render_output_dir": str(tmp_path / "renders"),
            "batch_size": 4,
            "max_concurrent_requests": 2,
            "rendering_modes": ["prim_only"],
            "fail_on_missing_prim_images": False,
        },
        object_store,
    )

    assert result["total_images_rendered"] == 1
    assert result["missing_image_prims"] == ["/World/B", "/World/C", "/World/D"]
    assert result["output_dir"] == str(tmp_path)
    assert result["failed_batches"] == [{"batch_start": 0, "render_mode": "prim_only"}]
    assert result["prim_data"][0]["images"] == [{"path": "World/A.png"}]
    object_store.set.assert_any_call(
        "stage_up_axis", str(UsdGeom.GetStageUpAxis(stage))
    )
    object_store.set.assert_any_call("prim_data", result["prim_data"])


@pytest.mark.asyncio
async def test_arun_accepts_explicit_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage, _ = _stage_with_mesh()
    task = USDPrimTraversalAndRenderingTask()
    object_store = Mock()
    object_store.get.side_effect = lambda key, default=None: {
        "usd_stage": stage,
        "rendering_backend": RemoteRenderingBackend(api_key="key"),
        "rendering_config": RenderingConfig(camera_ordering=["+x"]),
        "usd_model": None,
    }.get(key, default)

    def collect_with_recovered_guide(
        _stage: object,
        _filters: object,
        _listener: object,
        _diagnostics: object,
        _assembly_target_members: object,
        recovered_guide_gprim_targets: set[str],
    ) -> list[str]:
        recovered_guide_gprim_targets.add("/World/A")
        return ["/World/A"]

    monkeypatch.setattr(task, "_collect_prims", collect_with_recovered_guide)
    collect_and_filter = Mock(
        return_value=(
            ["/World/A"],
            [{"prim_path": "/World/A", "images": [], "metadata": {}}],
            0,
        )
    )
    monkeypatch.setattr(task, "_collect_and_filter_prims", collect_and_filter)
    monkeypatch.setattr(task, "_prepare_stages", Mock(return_value={"prim_only": {}}))

    async def fake_upload_stages_to_s3_async(*args: object, **kwargs: object):
        return []

    monkeypatch.setattr(
        task,
        "_upload_stages_to_s3_async",
        fake_upload_stages_to_s3_async,
    )
    monkeypatch.setattr(task, "_cleanup_s3", Mock())
    monkeypatch.setattr(task, "_check_blank_dataset_renders", Mock())

    async def fake_process_batch_async_with_retry_split(
        *args: object, **kwargs: object
    ):
        return (
            0,
            1,
            "prim_only",
            1,
            [],
            {"/World/A": [{"path": "explicit/World/A.png"}]},
        )

    monkeypatch.setattr(
        task,
        "_process_batch_async_with_retry_split",
        fake_process_batch_async_with_retry_split,
    )

    explicit_output = tmp_path / "explicit"
    context = {
        "render_output_dir": str(tmp_path / "renders"),
        "output_dir": str(explicit_output),
        "rendering_modes": ["prim_only"],
    }
    result = await task.arun(context, object_store)

    assert result["output_dir"] == str(explicit_output)
    propagated_config = collect_and_filter.call_args.args[3]
    assert propagated_config.recovered_guide_gprim_targets == ("/World/A",)
    assert context["recovered_guide_gprim_targets"] == ["/World/A"]


@pytest.mark.asyncio
async def test_upload_stages_to_s3_async_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    assert await task._upload_stages_to_s3_async({}, object(), {}, listener) == []

    calls: list[object] = []

    def fake_export_stage_to_s3(
        stage_arg: object, **kwargs: object
    ) -> tuple[str, str | None]:
        calls.append(stage_arg)
        if stage_arg == "plain":
            raise RuntimeError("upload failed")
        s3_uri = f"s3://bucket/{stage_arg}" if stage_arg != "single" else None
        return f"https://example.invalid/{stage_arg}", s3_uri

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.export_stage_to_s3",
        fake_export_stage_to_s3,
    )
    backend = RemoteRenderingBackend(api_key="key", use_data_uri=False)
    prepared = {
        "composition": {
            "type": "composition",
            "data": (("highlight", ["HC"], 2), ("plain", ["PC"], 2)),
        },
        "prim_only": {"type": "prim_only", "data": ("single", ["Cam"], 1)},
        "broken": {"type": "prim_only"},
    }

    cleanup = await task._upload_stages_to_s3_async(
        prepared,
        backend,
        {"usd_path": str(tmp_path / "scene.usda")},
        listener,
    )

    assert sorted(calls) == ["highlight", "plain", "single"]
    assert cleanup == [("s3://bucket/highlight", backend.s3_profile)]
    assert prepared["composition"]["highlight_url"].endswith("/highlight")
    assert "plain_url" not in prepared["composition"]
    assert prepared["prim_only"]["stage_url"].endswith("/single")


@pytest.mark.asyncio
async def test_process_batch_async_nonremote_and_remote_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    listener = Mock()
    config = RenderingConfig(
        image_width=2,
        use_background_color=True,
        background_color=(255, 255, 255),
    )

    monkeypatch.setattr(
        task,
        "_process_batch",
        Mock(return_value=(1, [], {"/World/A": [{"path": "sync.png"}]})),
    )
    assert await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {"prim_only": {"type": "prim_only", "config": config}},
        object(),
        "prim_only",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    ) == (0, 1, "prim_only", 1, [], {"/World/A": [{"path": "sync.png"}]})

    backend = RemoteRenderingBackend(api_key="key")
    assert await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {},
        backend,
        "missing",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    ) == (0, 1, "missing", 0, [], {})

    fallback = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "prim_only": {
                "type": "prim_only",
                "data": ("stage", ["Cam_posx"], 1),
                "config": config,
            }
        },
        backend,
        "prim_only",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )
    assert fallback == (0, 1, "prim_only", 1, [], {"/World/A": [{"path": "sync.png"}]})

    saved: list[tuple[object, Path]] = []

    async def fake_render_cameras_from_url(**kwargs: object) -> dict[str, object]:
        return {
            "results": [
                {
                    "camera": "Cam_posx",
                    "status": "success",
                    "blank_render_frames": [
                        {"frame": 0, "stats": {"blank": True, "reason": "solid"}}
                    ],
                    "images": [Image.new("RGBA", (2, 2), (0, 0, 255, 255))],
                    "sensors": {
                        "linear_depth": {
                            0: np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float32)
                        },
                        "ignored": {3: np.array([1], dtype=np.uint8)},
                    },
                }
            ]
        }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.render_cameras_from_url",
        fake_render_cameras_from_url,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.save_images_parallel",
        lambda save_tasks: saved.extend(save_tasks),
    )

    result = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "prim_only": {
                "type": "prim_only",
                "data": ("stage", ["Cam_posx"], 1),
                "config": config,
                "stage_url": "https://example.invalid/stage.usd",
            }
        },
        backend,
        "prim_only",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
        sensor_modes=["linear_depth"],
        image_height=2,
    )

    assert result[3] == 2
    assert {image["render_mode"] for image in result[5]["/World/A"]} == {
        "prim_only",
        "linear_depth",
    }
    assert result[5]["/World/A"][0]["blank_render"] is True
    assert len(saved) == 2

    unknown = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {"custom": {"type": "unknown", "config": config}},
        backend,
        "custom",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )
    assert unknown == (0, 1, "custom", 0, [], {})

    async def fail_render_cameras_from_url(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("remote failed")

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.render_cameras_from_url",
        fail_render_cameras_from_url,
    )
    failed = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "prim_only": {
                "type": "prim_only",
                "data": ("stage", ["Cam_posx"], 1),
                "config": config,
                "stage_url": "https://example.invalid/stage.usd",
            }
        },
        backend,
        "prim_only",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )
    assert failed[4][0]["error"] == "remote failed"


@pytest.mark.asyncio
async def test_process_batch_async_composition_url_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    backend = RemoteRenderingBackend(api_key="key")
    listener = Mock()
    saved: list[tuple[object, Path]] = []

    async def fake_render_composition_from_url(**kwargs: object):
        return (
            {
                "results": [
                    {
                        "camera": "Cam_posx",
                        "status": "success",
                        "images": [Image.new("RGB", (2, 2), "red")],
                    }
                ]
            },
            {
                "results": [
                    {
                        "camera": "Cam_posx",
                        "status": "success",
                        "images": [Image.new("RGB", (2, 2), "blue")],
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.render_composition_from_url",
        fake_render_composition_from_url,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.save_images_parallel",
        lambda save_tasks: saved.extend(save_tasks),
    )

    result = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "composition": {
                "type": "composition",
                "data": (("highlight", ["Cam_posx"], 1), ("plain", ["Cam_posx"], 1)),
                "config": RenderingConfig(
                    image_width=2,
                    enable_contour=False,
                    enable_bbox=False,
                ),
                "highlight_url": "https://example.invalid/highlight.usd",
                "plain_url": "https://example.invalid/plain.usd",
            }
        },
        backend,
        "composition",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )

    assert result[3] == 1
    assert result[5]["/World/A"][0]["render_mode"] == "composition"
    assert result[5]["/World/A"][0]["camera"] == "Cam_posx"
    assert len(saved) == 1

    monkeypatch.setattr(
        task,
        "_process_batch",
        Mock(return_value=(1, [], {"/World/A": [{"path": "fallback.png"}]})),
    )
    fallback = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "composition": {
                "type": "composition",
                "data": (("highlight", ["Cam_posx"], 1), ("plain", ["Cam_posx"], 1)),
                "config": RenderingConfig(image_width=2),
            }
        },
        backend,
        "composition",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )
    assert fallback == (
        0,
        1,
        "composition",
        1,
        [],
        {"/World/A": [{"path": "fallback.png"}]},
    )


@pytest.mark.asyncio
async def test_process_batch_async_composition_postprocess_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    backend = RemoteRenderingBackend(api_key="key")
    listener = Mock()
    saved: list[tuple[object, Path]] = []
    image_calls: list[str] = []

    async def fake_render_composition_from_url(**kwargs: object):
        return (
            {
                "results": [
                    {
                        "camera": "Cam_posx",
                        "status": "success",
                        "images": [
                            Image.new("RGBA", (2, 2), (255, 0, 0, 255)),
                            Image.new("RGBA", (2, 2), (0, 255, 0, 255)),
                        ],
                    },
                    {
                        "camera": "Cam_missing",
                        "status": "success",
                        "images": [Image.new("RGBA", (2, 2), (0, 0, 255, 255))],
                    },
                ]
            },
            {
                "results": [
                    {
                        "camera": "Cam_posx",
                        "status": "success",
                        "images": [Image.new("RGBA", (2, 2), (20, 20, 20, 255))],
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.render_composition_from_url",
        fake_render_composition_from_url,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.save_images_parallel",
        lambda save_tasks: saved.extend(save_tasks),
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering.paste_on_background",
        lambda image, color: image.convert("RGB"),
    )

    import world_understanding.utils.image_utils as image_utils

    monkeypatch.setattr(
        image_utils,
        "extract_non_black_outline",
        lambda image, **kwargs: image_calls.append("non_black")
        or Image.new("L", image.size, 255),
    )
    monkeypatch.setattr(
        image_utils,
        "draw_bounding_box_on_red",
        lambda image, **kwargs: image_calls.append("bbox")
        or Image.new("L", image.size, 255),
    )
    monkeypatch.setattr(
        image_utils,
        "paste_outline_to_image",
        lambda image, outline, color: image_calls.append(f"paste:{color}") or image,
    )

    result = await task._process_batch_async(
        0,
        2,
        ["/World/A", "/World/B"],
        {"/World/A": {"images": []}, "/World/B": {"images": []}},
        {
            "composition": {
                "type": "composition",
                "data": (("highlight", ["Cam_posx"], 2), ("plain", ["Cam_posx"], 2)),
                "config": RenderingConfig(
                    image_width=2,
                    use_background_color=True,
                    background_color=(255, 255, 255),
                    enable_contour=True,
                    contour_method="non_black",
                    enable_bbox=True,
                ),
                "highlight_url": "https://example.invalid/highlight.usd",
                "plain_url": "https://example.invalid/plain.usd",
            }
        },
        backend,
        "composition",
        tmp_path,
        tmp_path,
        1,
        2,
        listener,
    )

    assert result[3] == 3
    assert result[5]["/World/A"][0]["render_mode"] == "composition"
    assert len(saved) == 3
    assert "non_black" in image_calls
    assert "bbox" in image_calls


@pytest.mark.asyncio
async def test_process_batch_async_composition_red_contour_and_occlusion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    backend = RemoteRenderingBackend(api_key="key")
    listener = Mock()
    saved: list[tuple[object, Path]] = []
    image_calls: list[str] = []

    async def fake_render_composition_from_url(**kwargs: object):
        return (
            {
                "results": [
                    {
                        "camera": "Cam_posx",
                        "status": "success",
                        "images": [Image.new("RGB", (2, 2), "red")],
                    }
                ]
            },
            {
                "results": [
                    {
                        "camera": "Cam_posx",
                        "status": "success",
                        "images": [Image.new("RGB", (2, 2), "blue")],
                    }
                ]
            },
        )

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.render_composition_from_url",
        fake_render_composition_from_url,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.save_images_parallel",
        lambda save_tasks: saved.extend(save_tasks),
    )

    import world_understanding.utils.image_utils as image_utils

    monkeypatch.setattr(
        image_utils,
        "extract_red_outline",
        lambda image, **kwargs: image_calls.append("red")
        or Image.new("L", image.size, 255),
    )
    monkeypatch.setattr(
        image_utils,
        "paste_outline_to_image",
        lambda image, outline, color: image_calls.append(f"paste:{color}") or image,
    )
    monkeypatch.setattr(
        image_utils, "is_prim_visible_in_image", lambda *args, **kwargs: False
    )

    result = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "composition": {
                "type": "composition",
                "data": (("highlight", ["Cam_posx"], 1), ("plain", ["Cam_posx"], 1)),
                "config": RenderingConfig(
                    image_width=2,
                    enable_contour=True,
                    contour_method="red",
                    enable_bbox=False,
                    skip_occluded_images=True,
                ),
                "highlight_url": "https://example.invalid/highlight.usd",
                "plain_url": "https://example.invalid/plain.usd",
            }
        },
        backend,
        "composition",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )

    assert result[3] == 0
    assert saved == []
    assert result[5] == {}

    monkeypatch.setattr(
        image_utils, "is_prim_visible_in_image", lambda *args, **kwargs: True
    )
    visible = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "composition": {
                "type": "composition",
                "data": (("highlight", ["Cam_posx"], 1), ("plain", ["Cam_posx"], 1)),
                "config": RenderingConfig(
                    image_width=2,
                    enable_contour=True,
                    contour_method="red",
                    enable_bbox=False,
                    skip_occluded_images=True,
                ),
                "highlight_url": "https://example.invalid/highlight.usd",
                "plain_url": "https://example.invalid/plain.usd",
            }
        },
        backend,
        "composition",
        tmp_path,
        tmp_path,
        1,
        1,
        listener,
    )

    assert visible[3] == 1
    assert "red" in image_calls


@pytest.mark.asyncio
async def test_process_batch_async_single_stage_occlusion_sensors_and_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    backend = RemoteRenderingBackend(api_key="key")
    listener = Mock()
    saved: list[tuple[object, Path]] = []

    async def fake_render_cameras_from_url(**kwargs: object) -> dict[str, object]:
        return {
            "results": [
                {
                    "camera": "Front",
                    "status": "success",
                    "images": [
                        Image.new("RGB", (2, 2), "red"),
                        Image.new("RGB", (2, 2), "green"),
                        Image.new("RGB", (2, 2), "blue"),
                        Image.new("RGB", (2, 2), "white"),
                    ],
                    "sensors": {
                        "linear_depth": {
                            0: np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float32),
                            1: np.array([1, 2, 3, 4, 5], dtype=np.uint8),
                            2: np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                        }
                    },
                }
            ]
        }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.render_cameras_from_url",
        fake_render_cameras_from_url,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.save_images_parallel",
        lambda save_tasks: saved.extend(save_tasks),
    )

    import world_understanding.utils.image_utils as image_utils

    visibility = iter([False, True, True])
    monkeypatch.setattr(
        image_utils,
        "is_prim_visible_in_image",
        lambda *args, **kwargs: next(visibility),
    )

    result = await task._process_batch_async(
        0,
        3,
        ["/World/A", "/World/B", "/World/Missing"],
        {"/World/A": {"images": []}, "/World/B": {"images": []}},
        {
            "prim_with_stage": {
                "type": "prim_with_stage",
                "data": ("stage", ["Front"], 3),
                "config": RenderingConfig(
                    image_width=2,
                    should_highlight_prim=True,
                    should_render_prim_only=False,
                    skip_occluded_images=True,
                ),
                "stage_url": "https://example.invalid/stage.usd",
            }
        },
        backend,
        "prim_with_stage",
        tmp_path,
        tmp_path,
        1,
        3,
        listener,
        sensor_modes=["linear_depth"],
        image_height=2,
    )

    assert result[3] == 2
    assert {image["render_mode"] for image in result[5]["/World/A"]} == {"linear_depth"}
    assert result[5]["/World/B"][0]["render_mode"] == "prim_with_stage"
    assert len(saved) == 2


@pytest.mark.asyncio
async def test_process_batch_async_custom_render_mode_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = USDPrimTraversalAndRenderingTask()
    backend = RemoteRenderingBackend(api_key="key")
    saved: list[tuple[object, Path]] = []

    async def fake_render_cameras_from_url(**kwargs: object) -> dict[str, object]:
        return {
            "results": [
                {
                    "camera": "Cam_posx",
                    "status": "success",
                    "images": [Image.new("RGB", (2, 2), "green")],
                }
            ]
        }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.render_cameras_from_url",
        fake_render_cameras_from_url,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.save_images_parallel",
        lambda save_tasks: saved.extend(save_tasks),
    )

    result = await task._process_batch_async(
        0,
        1,
        ["/World/A"],
        {"/World/A": {"images": []}},
        {
            "custom": {
                "type": "prim_only",
                "data": ("stage", ["Cam_posx"], 1),
                "config": RenderingConfig(image_width=2),
                "stage_url": "https://example.invalid/stage.usd",
            }
        },
        backend,
        "custom",
        tmp_path,
        tmp_path,
        1,
        1,
        Mock(),
    )

    assert result[3] == 1
    assert result[5]["/World/A"][0]["path"].endswith("_custom.png")
