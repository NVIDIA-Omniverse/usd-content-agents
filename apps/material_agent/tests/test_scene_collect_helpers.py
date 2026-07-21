# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for scene.collect helper functions."""

from __future__ import annotations

import base64
import json
import logging
import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade

from material_agent.scene.collect import (
    _build_cascaded_payload_map,
    _camera_config_number,
    _collect_mesh_paths_from_layer,
    _collect_mesh_paths_from_stage,
    _copy_materials_from_library,
    _extract_material_name,
    _fill_prediction_gaps,
    _find_predictions_path,
    _load_material_library,
    _load_payload_predictions,
    _material_for_source_mesh,
    _merge_predictions,
    _mesh_attr_len,
    _mesh_structural_fingerprint,
    _ordered_mesh_fingerprint_mismatches,
    _path_to_filename,
    _PathIndex,
    _propagate_instance_bindings,
    _prune_nested_prefixes,
    _relative_path_under_any_member,
    _remap_asset_paths_in_prim,
    _remap_single_asset_path,
    _target_path_exists_or_payload_backed,
    _write_binding_over,
    _write_dominant_mesh_bindings,
    _write_material_bindings,
    _write_ordered_mesh_bindings,
    author_projected_material_layer,
    render_composed_scene,
)
from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)


def _write_jsonl(path: Path, lines: list[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line) + "\n")
    return path


def _create_layer(path: Path, *, sublayers: list[str] | None = None) -> Path:
    layer = Sdf.Layer.CreateNew(str(path))
    if sublayers is not None:
        layer.subLayerPaths = sublayers
    layer.Save()
    return path


def _define_mesh_with_points(stage: Usd.Stage, path: str, point_count: int) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(float(i), 0.0, 0.0) for i in range(point_count)])


def _create_renderable_scene(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()
    return path


def test_author_projected_material_layer_accepts_composed_arc_children(
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "materials.usda"
    library_stage = Usd.Stage.CreateNew(str(library_path))
    library_root = UsdGeom.Xform.Define(library_stage, "/World")
    library_stage.SetDefaultPrim(library_root.GetPrim())
    UsdShade.Material.Define(library_stage, "/World/Looks/Plastic_Red")
    library_stage.GetRootLayer().Save()
    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        """library_path: materials.usda
entries:
  - name: Plastic Red
    binding: /World/Looks/Plastic_Red
""",
        encoding="utf-8",
    )

    asset_path = tmp_path / "asset.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    asset_root = UsdGeom.Xform.Define(asset_stage, "/Asset")
    asset_stage.SetDefaultPrim(asset_root.GetPrim())
    UsdGeom.Cube.Define(asset_stage, "/Asset/Part")
    asset_stage.GetRootLayer().Save()

    source_path = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/World")
    source_stage.SetDefaultPrim(source_root.GetPrim())
    source_material = UsdShade.Material.Define(
        source_stage, "/World/Looks/SourceMaterial"
    )
    referenced = UsdGeom.Xform.Define(source_stage, "/World/Referenced").GetPrim()
    referenced.GetReferences().AddReference(str(asset_path), "/Asset")
    UsdShade.MaterialBindingAPI.Apply(referenced).Bind(
        source_material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )
    payload = UsdGeom.Xform.Define(source_stage, "/World/Payload").GetPrim()
    payload.GetPayloads().AddPayload(str(asset_path), "/Asset")
    source_stage.GetRootLayer().Save()
    target_path = "/World/Referenced/Part"
    payload_target_path = "/World/Payload/Part"
    assert source_stage.GetRootLayer().GetPrimAtPath(target_path) is None
    assert source_stage.GetRootLayer().GetPrimAtPath(payload_target_path) is None

    output_path = tmp_path / "material_layer.usda"
    summary = author_projected_material_layer(
        source_path,
        output_path,
        library_yaml,
        {
            target_path: "Plastic Red",
            payload_target_path: "Plastic Red",
        },
    )

    output_layer = Sdf.Layer.FindOrOpen(str(output_path))
    assert output_layer is not None
    target_spec = output_layer.GetPrimAtPath(target_path)
    assert target_spec is not None
    assert target_spec.relationships[
        "material:binding"
    ].targetPathList.explicitItems == [
        Sdf.Path(summary["material_paths"]["Plastic Red"])
    ]
    assert output_layer.GetPrimAtPath(payload_target_path) is not None
    assert summary["weakened_ancestor_binding_count"] == 1

    composition_path = tmp_path / "composition.usda"
    composition_layer = Sdf.Layer.CreateNew(str(composition_path))
    composition_layer.subLayerPaths = [str(output_path), str(source_path)]
    composition_layer.Save()
    composed_stage = Usd.Stage.Open(str(composition_path))
    bound_material, _relationship = UsdShade.MaterialBindingAPI(
        composed_stage.GetPrimAtPath(target_path)
    ).ComputeBoundMaterial()
    assert bound_material.GetPath() == Sdf.Path(
        summary["material_paths"]["Plastic Red"]
    )
    payload_bound_material, _payload_relationship = UsdShade.MaterialBindingAPI(
        composed_stage.GetPrimAtPath(payload_target_path)
    ).ComputeBoundMaterial()
    assert payload_bound_material.GetPath() == Sdf.Path(
        summary["material_paths"]["Plastic Red"]
    )


def _encoded_png(mode: str = "RGB") -> str:
    image = Image.new(mode, (4, 4), (32, 64, 96))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_path_index_and_prefix_helpers_cover_edge_cases() -> None:
    index = _PathIndex(
        {
            "/Root/Asset": "Steel",
            "/Root/Asset/Mesh": "Steel",
            "/Root/Other": "Plastic",
        }
    )

    assert index.get_paths_under("/Root/Asset") == {
        "/Root/Asset": "Steel",
        "/Root/Asset/Mesh": "Steel",
    }
    assert index.is_under_any("/Root/Asset/Mesh", []) is False
    assert index.is_under_any("/Root/Asset/Mesh", ["/Root/Asset"]) is True
    assert _prune_nested_prefixes(
        {"/Root/Asset", "/Root/Asset/Mesh", "/Root/Other"}
    ) == ["/Root/Asset", "/Root/Other"]
    assert _relative_path_under_any_member("/Root/Asset", ["/Root/Asset"]) == ""
    assert (
        _relative_path_under_any_member("/Root/Asset/Mesh", ["/Root/Asset"]) == "/Mesh"
    )
    assert _relative_path_under_any_member("/Root/Missing", ["/Root/Asset"]) is None


def test_material_library_and_prediction_edge_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not\n- a\n- mapping\n")
    assert _load_material_library(list_yaml) == (None, {})

    bad_materials_yaml = tmp_path / "bad-materials.yaml"
    bad_materials_yaml.write_text("materials:\n  - nope\n")
    assert _load_material_library(bad_materials_yaml) == (None, {})

    bad_entries_yaml = tmp_path / "bad-entries.yaml"
    bad_entries_yaml.write_text("library_path: library.usda\nentries: nope\n")
    resolved_library, name_to_prim = _load_material_library(bad_entries_yaml)
    assert resolved_library == (tmp_path / "library.usda").resolve()
    assert name_to_prim == {}

    assert (
        _find_predictions_path(SubAsset(id="none", name="None", prim_path="/Root/None"))
        is None
    )

    payload_predictions = _write_jsonl(
        tmp_path / "payload.jsonl",
        ["", "{bad-json", {"id": "/Root/Payload", "materials": "Steel"}],
    )
    blank_predictions = _write_jsonl(
        tmp_path / "blank.jsonl",
        ["", "{bad-json", {"id": "", "materials": "Plastic"}],
    )
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="missing",
                name="MissingPredictions",
                prim_path="/Root/Missing",
                status="completed",
            ),
            SubAsset(
                id="blank",
                name="BlankPredictions",
                prim_path="/Root/Blank",
                predictions_path=str(blank_predictions),
                status="completed",
            ),
        ],
        payload_groups=[
            PayloadGroup(
                id="payload-missing",
                group_name="PayloadMissing",
                payload_file="payload.usd",
                status="completed",
            ),
            PayloadGroup(
                id="payload",
                group_name="Payload",
                payload_file="payload.usd",
                predictions_path=str(payload_predictions),
                status="completed",
            ),
        ],
    )

    merged = _merge_predictions(manifest)

    assert merged == {"/Root/Payload": "Steel"}

    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(lambda *_args: None))
    original = {"/Root/Asset": "Steel"}
    assert (
        _fill_prediction_gaps(tmp_path / "missing.usda", original, manifest) is original
    )


def test_mesh_and_target_helper_edge_cases(tmp_path: Path) -> None:
    layer = Sdf.Layer.CreateNew(str(tmp_path / "layer.usda"))
    assert _collect_mesh_paths_from_layer(layer, "/Missing") == []
    mesh_spec = Sdf.CreatePrimInLayer(layer, "/Root/MeshInLayer")
    mesh_spec.typeName = "Mesh"
    layer.Save()
    assert _collect_mesh_paths_from_layer(layer, "/Root") == ["/Root/MeshInLayer"]

    stage_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    _define_mesh_with_points(stage, "/Root/MeshA", 3)
    _define_mesh_with_points(stage, "/Root/MeshB", 4)
    stage.Save()

    assert _collect_mesh_paths_from_stage(stage, "/Missing") == []
    assert _material_for_source_mesh("/Root/MeshA", {"/Root": "Ancestor"}) == "Ancestor"
    assert (
        _material_for_source_mesh("/Root/MeshA", {"/Root/MeshA/Subset": "Child"})
        == "Child"
    )
    assert _material_for_source_mesh("/Root/Group/MeshA", {"/Other": "Steel"}) is None
    assert _material_for_source_mesh("/Root/MeshA", {"/Other": "Steel"}) is None
    assert _material_for_source_mesh("/Root/MeshA", {}) is None

    class RaisingAttr:
        def Get(self) -> object:
            raise RuntimeError("bad attr")

    class ScalarAttr:
        def __init__(self, value: object) -> None:
            self.value = value

        def Get(self) -> object:
            return self.value

    assert _mesh_attr_len(RaisingAttr()) is None
    assert _mesh_attr_len(ScalarAttr(None)) is None
    assert _mesh_attr_len(ScalarAttr(5)) == 1
    assert _mesh_structural_fingerprint(None, "/Root/MeshA") is None
    assert _mesh_structural_fingerprint(stage, "/Root") is None
    assert _ordered_mesh_fingerprint_mismatches(None, ["/A"], ["/B"]) == ([], 0)
    mismatches, checked = _ordered_mesh_fingerprint_mismatches(
        stage, ["/Root/MeshA"], ["/Root/MeshB"]
    )
    assert checked == 1
    assert mismatches

    out_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))
    assert (
        _write_ordered_mesh_bindings(
            layer=out_layer,
            source_meshes=[],
            target_meshes=["/Root/MeshA"],
            source_bindings={"/Root/MeshA": "Steel"},
            name_to_prim={"Steel": "/Root/Looks/Steel"},
        )
        == 0
    )
    assert (
        _write_dominant_mesh_bindings(
            layer=out_layer,
            target_meshes=["/Root/MeshA"],
            source_bindings={},
            name_to_prim={"Steel": "/Root/Looks/Steel"},
            scene_layer=None,
            skip_targets=set(),
            reason="empty",
        )
        == 0
    )
    assert (
        _write_dominant_mesh_bindings(
            layer=out_layer,
            target_meshes=["/Root/MeshA"],
            source_bindings={"/Root/MeshA": "Missing"},
            name_to_prim={},
            scene_layer=None,
            skip_targets=set(),
            reason="missing material",
        )
        == 0
    )
    assert (
        _write_dominant_mesh_bindings(
            layer=out_layer,
            target_meshes=["/Root/MeshA"],
            source_bindings={"/Root/MeshA": "Steel"},
            name_to_prim={"Steel": "/Root/Looks/Steel"},
            scene_layer=None,
            skip_targets={"/Root/MeshA"},
            reason="skipped",
        )
        == 0
    )
    _write_binding_over(out_layer, "/Root/MeshA", "/Root/Looks/Steel")
    assert (
        _write_dominant_mesh_bindings(
            layer=out_layer,
            target_meshes=["/Root/MeshA"],
            source_bindings={"/Root/MeshA": "Steel"},
            name_to_prim={"Steel": "/Root/Looks/Steel"},
            scene_layer=None,
            skip_targets=set(),
            reason="existing",
        )
        == 0
    )


def test_binding_and_material_copy_failure_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))

    assert (
        _write_material_bindings(
            out_layer,
            {"/Root/MissingMaterial": "Ghost"},
            {"Steel": "/Root/Looks/Steel"},
        )
        == 0
    )

    original_create = Sdf.CreatePrimInLayer
    monkeypatch.setattr(Sdf, "CreatePrimInLayer", lambda *_args: None)
    assert _write_binding_over(out_layer, "/Root/NoSpec", "/Root/Looks/Steel") == 0
    assert (
        _write_material_bindings(
            out_layer,
            {"/Root/NoSpec": "Steel"},
            {"Steel": "/Root/Looks/Steel"},
        )
        == 0
    )
    monkeypatch.setattr(Sdf, "CreatePrimInLayer", original_create)

    scene_layer = Sdf.Layer.CreateNew(str(tmp_path / "scene-with-subset.usda"))
    mesh = Sdf.CreatePrimInLayer(scene_layer, "/Root/Mesh")
    mesh.typeName = "Mesh"
    subset = Sdf.CreatePrimInLayer(scene_layer, "/Root/Mesh/Diffuse_0")
    subset.typeName = "GeomSubset"
    scene_layer.Save()

    calls = 0

    def flaky_create(layer: Sdf.Layer, prim_path: str):
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return original_create(layer, prim_path)

    monkeypatch.setattr(Sdf, "CreatePrimInLayer", flaky_create)
    assert (
        _write_binding_over(
            out_layer,
            "/Root/Mesh",
            "/Root/Looks/Steel",
            scene_layer=scene_layer,
        )
        == 1
    )
    monkeypatch.setattr(Sdf, "CreatePrimInLayer", original_create)

    assert (
        _copy_materials_from_library(
            Sdf.Layer.CreateAnonymous(),
            tmp_path / "missing-library.usda",
            {"Steel": "/World/Looks/Steel"},
            tmp_path / "composed.usda",
        )
        == {}
    )

    library_usd = tmp_path / "library.usda"
    library_layer = Sdf.Layer.CreateNew(str(library_usd))
    library_layer.defaultPrim = "World"
    Sdf.CreatePrimInLayer(library_layer, "/World")
    Sdf.CreatePrimInLayer(library_layer, "/World/Looks/Steel")
    library_layer.Save()

    assert _copy_materials_from_library(
        Sdf.Layer.CreateAnonymous(),
        library_usd,
        {"WorldRoot": "/World"},
        tmp_path / "composed.usda",
        scene_default_prim="Root",
    ) == {"/World": "/Root"}

    original_copy = Sdf.CopySpec
    monkeypatch.setattr(Sdf, "CopySpec", lambda *_args: False)
    assert (
        _copy_materials_from_library(
            Sdf.Layer.CreateAnonymous(),
            library_usd,
            {"Steel": "/World/Looks/Steel"},
            tmp_path / "composed.usda",
        )
        == {}
    )
    monkeypatch.setattr(Sdf, "CopySpec", original_copy)

    _remap_asset_paths_in_prim(
        out_layer,
        Sdf.Path("/Root/NoSuchPrim"),
        tmp_path,
        tmp_path / "output",
    )

    monkeypatch.setattr(
        "material_agent.scene.collect.os.path.relpath",
        lambda *_args: (_ for _ in ()).throw(ValueError("different drive")),
    )
    assert (
        _remap_single_asset_path("textures/a.png", tmp_path, tmp_path / "out")
        == (tmp_path / "textures" / "a.png").resolve().as_posix()
    )


def test_ordered_mesh_binding_edge_branches(tmp_path: Path) -> None:
    layer = Sdf.Layer.CreateNew(str(tmp_path / "ordered.usda"))
    assert (
        _write_ordered_mesh_bindings(
            layer=layer,
            source_meshes=["/Root/SourceA", "/Root/SourceB", "/Root/SourceC"],
            target_meshes=["/Root/TargetA", "/Root/TargetB", "/Root/TargetC"],
            source_bindings={
                "/Root/SourceA": "Steel",
                "/Root/SourceC": "Ghost",
            },
            name_to_prim={"Steel": "/Root/Looks/Steel"},
            skip_targets={"/Root/TargetA"},
        )
        == 0
    )

    scene_path = tmp_path / "fingerprints.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    _define_mesh_with_points(stage, "/Root/SourceMesh", 3)
    _define_mesh_with_points(stage, "/Root/TargetMesh", 3)
    stage.Save()

    assert (
        _write_ordered_mesh_bindings(
            layer=layer,
            source_meshes=["/Root/SourceMesh"],
            target_meshes=["/Root/TargetMesh"],
            source_bindings={"/Root/SourceMesh": "Steel"},
            name_to_prim={"Steel": "/Root/Looks/Steel"},
            scene_stage=stage,
        )
        == 1
    )


def test_camera_config_number_warns_for_non_numeric_value(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="material_agent.scene.collect")

    assert _camera_config_number({"dome_light": object()}, "dome_light") is None
    assert "Ignoring non-numeric camera config value" in caplog.text


def test_target_path_exists_or_payload_backed_edge_cases() -> None:
    class FakePrim:
        def __init__(
            self,
            valid: bool,
            *,
            has_payload: bool = False,
            has_authored_payloads: bool = False,
        ) -> None:
            self._valid = valid
            self._has_payload = has_payload
            self._has_authored_payloads = has_authored_payloads

        def IsValid(self) -> bool:
            return self._valid

        def HasPayload(self) -> bool:
            return self._has_payload

        def HasAuthoredPayloads(self) -> bool:
            return self._has_authored_payloads

    class FakeStage:
        def __init__(self, prims: dict[str, FakePrim]) -> None:
            self.prims = prims

        def GetPrimAtPath(self, path: str) -> FakePrim | None:
            return self.prims.get(path)

    assert _target_path_exists_or_payload_backed(
        FakeStage({"/Root/Member/Mesh": FakePrim(True)}),
        "/Root/Member/Mesh",
        "/Root/Member",
    )
    assert _target_path_exists_or_payload_backed(
        FakeStage({"/Root/Member": FakePrim(True, has_payload=True)}),
        "/Root/Member/Missing",
        "/Root/Member",
    )
    assert _target_path_exists_or_payload_backed(
        FakeStage({"/Root/Member": FakePrim(True, has_authored_payloads=True)}),
        "/Root/Member/Missing",
        "/Root/Member",
    )
    assert not _target_path_exists_or_payload_backed(
        FakeStage({"/Root/Member": FakePrim(True)}),
        "/Root/Member/Missing",
        "/Root/Member",
    )


def test_find_predictions_path_fallback_order(tmp_path: Path) -> None:
    working_dir = tmp_path / "asset"
    restored = _write_jsonl(
        working_dir / "restored" / "restored_predictions.jsonl",
        [{"id": "/Root/A", "materials": "Steel"}],
    )
    manifest_path = _write_jsonl(
        tmp_path / "manifest_predictions.jsonl",
        [{"id": "/Root/B", "materials": "Copper"}],
    )
    raw = _write_jsonl(
        working_dir / "predictions" / "predictions.jsonl",
        [{"id": "/Root/C", "materials": "Plastic"}],
    )

    asset = SubAsset(
        id="a1",
        name="Asset",
        prim_path="/Root/Asset",
        working_dir=str(working_dir),
        predictions_path=str(manifest_path),
        status="completed",
    )
    assert _find_predictions_path(asset) == restored

    restored.unlink()
    assert _find_predictions_path(asset) == manifest_path

    manifest_path.unlink()
    assert _find_predictions_path(asset) == raw


def test_render_composed_scene_uses_explicit_camera_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scene_path = _create_renderable_scene(tmp_path / "scene.usda")
    captured: dict[str, object] = {}

    class FakeRenderingBackend:
        def render(
            self,
            *,
            stage: Usd.Stage,
            cameras: list[str],
            image_width: int,
            image_height: int,
            frames: str,
        ) -> dict[str, object]:
            captured["cameras"] = cameras
            captured["image_width"] = image_width
            captured["image_height"] = image_height
            captured["frames"] = frames
            captured["has_camera"] = stage.GetPrimAtPath(cameras[0]).IsValid()
            captured["has_dome"] = stage.GetPrimAtPath(
                "/RenderLights/DomeLight"
            ).IsValid()
            captured["has_distant"] = stage.GetPrimAtPath(
                "/RenderLights/DistantLight"
            ).IsValid()
            return {
                "successful_cameras": 1,
                "results": [
                    {
                        "images": [
                            Image.new("RGBA", (4, 4), (32, 64, 96, 255)),
                        ]
                    }
                ],
            }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering.RemoteRenderingBackend",
        FakeRenderingBackend,
    )

    rendered = render_composed_scene(
        composed_usd_path=scene_path,
        output_dir=tmp_path / "renders",
        camera_config={
            "name": "scene_humanoid_camera",
            "focal_length_mm": 130.0,
            "cam_x": 5607.0,
            "cam_y": -2049.0,
            "cam_z": 150.0,
            "target_z": -200.0,
            "dome_light": 1500.0,
            "distant_light": 800.0,
            "image_width": 320,
            "image_height": 180,
        },
    )

    assert rendered == [
        tmp_path / "renders" / "composed_scene_scene_humanoid_camera.png"
    ]
    assert captured["cameras"] == ["/Cameras/SceneCamera_configured"]
    assert captured["image_width"] == 320
    assert captured["image_height"] == 180
    assert captured["frames"] == "0"
    assert captured["has_camera"] is True
    assert captured["has_dome"] is True
    assert captured["has_distant"] is True


def test_render_composed_scene_rejects_non_positive_camera_dimensions(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        render_composed_scene(
            composed_usd_path=tmp_path / "missing.usda",
            output_dir=tmp_path / "renders",
            camera_config={"image_width": 320, "image_height": 0},
        )


def test_render_composed_scene_clear_materials_default_cameras_and_base64_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_scene = _create_renderable_scene(tmp_path / "base.usda")
    material_layer = _create_layer(tmp_path / "materials.usda")
    composed_layer = Sdf.Layer.CreateNew(str(tmp_path / "composed.usda"))
    composed_layer.subLayerPaths = [str(material_layer), str(base_scene)]
    composed_layer.defaultPrim = "Root"
    composed_layer.Save()
    calls: list[dict[str, object]] = []

    class FakeRenderingBackend:
        def render(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "successful_cameras": 1,
                "results": [{"images": [{"image": _encoded_png("RGB")}]}],
            }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering.RemoteRenderingBackend",
        FakeRenderingBackend,
    )
    monkeypatch.setattr(
        "world_understanding.utils.usd.prim.nullify_materials",
        lambda stage: None,
    )

    rendered = render_composed_scene(
        composed_usd_path=tmp_path / "composed.usda",
        output_dir=tmp_path / "renders",
        image_width=64,
        image_height=128,
        clear_materials=True,
    )

    assert len(rendered) == 2
    assert {path.name for path in rendered} == {
        "composed_scene_posx_posy_posz.png",
        "composed_scene_negx_negy_negz.png",
    }
    assert all(path.exists() for path in rendered)
    assert [call["image_width"] for call in calls] == [64, 64]
    assert all(call["image_height"] == 128 for call in calls)


def test_render_composed_scene_logs_failed_render_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scene_path = _create_renderable_scene(tmp_path / "scene.usda")
    responses: list[object] = [
        {"successful_cameras": 1, "results": [{"images": [object()]}]},
        {"successful_cameras": 1, "results": [{"images": []}]},
        {"successful_cameras": 0, "results": [{"error": "backend failed"}]},
        RuntimeError("boom"),
    ]

    class FakeRenderingBackend:
        def render(self, **_kwargs: object) -> dict[str, object]:
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering.RemoteRenderingBackend",
        FakeRenderingBackend,
    )

    rendered = render_composed_scene(
        composed_usd_path=scene_path,
        output_dir=tmp_path / "renders",
        camera_corners=["+x", "+y", "+z", "-x"],
    )

    assert rendered == []
    assert responses == []


def test_extract_material_name_supports_supported_prediction_shapes() -> None:
    assert _extract_material_name({"materials": {"material": "Steel"}}) == "Steel"
    assert _extract_material_name({"materials": "Copper"}) == "Copper"
    assert _extract_material_name({"material": "Plastic"}) == "Plastic"
    assert _extract_material_name({"materials": {"other": "x"}}) is None
    assert _extract_material_name({"materials": 123}) is None


def test_merge_predictions_prefers_restored_predictions_and_infers_parent(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "asset_a"
    _write_jsonl(
        working_dir / "predictions" / "predictions.jsonl",
        [{"id": "/Root/Asset/Mesh/Diffuse_0", "materials": "Wrong"}],
    )
    _write_jsonl(
        working_dir / "restored" / "restored_predictions.jsonl",
        [
            {
                "id": "/Root/Asset/Mesh/Diffuse_0",
                "materials": {"material": "Steel"},
            },
            {
                "id": "/Root/Asset/Mesh/Diffuse_1",
                "materials": {"material": "Steel"},
            },
            "{not-json",
        ],
    )
    payload_predictions = _write_jsonl(
        tmp_path / "payload_predictions.jsonl",
        [{"id": "/Root/Payload/Mesh", "materials": "Plastic"}],
    )

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="asset-a",
                name="AssetA",
                prim_path="/Root/Asset",
                working_dir=str(working_dir),
                predictions_path=str(tmp_path / "unused.jsonl"),
                status="completed",
            ),
            SubAsset(
                id="asset-b",
                name="Skipped",
                prim_path="/Root/Skipped",
                status="pending",
            ),
        ],
        payload_groups=[
            PayloadGroup(
                id="payload-a",
                group_name="PayloadA",
                payload_file=str(tmp_path / "payload_a.usda"),
                predictions_path=str(payload_predictions),
                status="completed",
            )
        ],
    )

    merged = _merge_predictions(manifest)

    assert merged["/Root/Asset/Mesh/Diffuse_0"] == "Steel"
    assert merged["/Root/Asset/Mesh/Diffuse_1"] == "Steel"
    assert merged["/Root/Asset/Mesh"] == "Steel"
    assert merged["/Root/Payload/Mesh"] == "Plastic"
    assert "/Root/Skipped" not in merged


def test_fill_prediction_gaps_uses_sibling_majority_then_asset_dominant(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/Asset")
    UsdGeom.Xform.Define(stage, "/Root/Asset/Group")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Group/MeshA")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Group/MeshB")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Group/MeshC")
    UsdGeom.Xform.Define(stage, "/Root/Asset/Other")
    UsdGeom.Mesh.Define(stage, "/Root/Asset/Other/MeshD")
    stage.Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="asset-a",
                name="Asset",
                prim_path="/Root/Asset",
                status="completed",
            )
        ]
    )
    prim_to_material = {
        "/Root/Asset/Group/MeshA": "Steel",
        "/Root/Asset/Group/MeshB": "Steel",
    }

    filled = _fill_prediction_gaps(scene_path, prim_to_material, manifest)

    assert filled["/Root/Asset/Group/MeshC"] == "Steel"
    assert filled["/Root/Asset/Other/MeshD"] == "Steel"


def test_suffix_gap_fill_targets_representative_relative_member_subtree(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    source_mesh = "/Root/MemberA/Bolt/Shared/shape/mesh"
    target_mesh = "/Root/MemberB/Bolt/Shared/shape/mesh"
    unrelated_mesh = "/Root/MemberB/Huge/Bolt/Shared/shape/mesh"
    UsdGeom.Mesh.Define(stage, source_mesh)
    UsdGeom.Mesh.Define(stage, target_mesh)
    UsdGeom.Mesh.Define(stage, unrelated_mesh)
    stage.Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="bolt-rep",
                name="Bolt",
                prim_path="/Root/MemberA/Bolt",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="member_group",
                representative_id="bolt-rep",
                member_paths=["/Root/MemberA", "/Root/MemberB"],
            )
        ],
    )

    filled = _fill_prediction_gaps(
        scene_path,
        {source_mesh: "Steel"},
        manifest,
    )

    assert filled[target_mesh] == "Steel"
    assert unrelated_mesh not in filled


def test_fill_prediction_gaps_covers_missing_roots_and_short_suffixes(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/Asset")
    for name in ["A", "B", "C", "D"]:
        UsdGeom.Mesh.Define(stage, f"/Root/Asset/Group/Mesh{name}")
    UsdGeom.Xform.Define(stage, "/Root/Short")
    UsdGeom.Mesh.Define(stage, "/Root/Short/Mesh")
    long_source = "/Root/Source/A/B/C/D/E/F"
    UsdGeom.Mesh.Define(stage, long_source)
    stage.Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="asset",
                name="Asset",
                prim_path="/Root/Asset",
                status="completed",
            ),
            SubAsset(
                id="missing",
                name="Missing",
                prim_path="/Root/Missing",
                status="completed",
            ),
            SubAsset(
                id="short",
                name="Short",
                prim_path="/Root/Short",
                status="completed",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="unknown_rep",
                representative_id="not-selected",
                member_paths=["/Root/Short"],
            )
        ],
    )

    filled = _fill_prediction_gaps(
        scene_path,
        {
            "/Root/Asset/Group/MeshA": "Steel",
            "/Root/Asset/Group/MeshB": "Plastic",
            "/Root/Asset/Group/MeshC": "Rubber",
            long_source: "Steel",
        },
        manifest,
    )

    assert filled["/Root/Asset/Group/MeshD"] in {"Steel", "Plastic", "Rubber"}
    assert "/Root/Short/Mesh" not in filled


def test_fill_prediction_gaps_handles_defensive_empty_representative_path(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    source = "/Root/Fake/A/B/C/D/E/F"
    UsdGeom.Mesh.Define(stage, source)
    stage.Save()

    class FlakyAsset:
        id = "fake"
        name = "Fake"
        status = "completed"

        def __init__(self) -> None:
            self.calls = 0

        @property
        def prim_path(self) -> str:
            self.calls += 1
            return "/Root/Fake" if self.calls <= 2 else ""

    class FakeManifest:
        instance_groups = [
            InstanceGroup(
                group_name="fake_group",
                representative_id="fake",
                member_paths=["/Root/Fake"],
            )
        ]

        def __init__(self) -> None:
            self.asset = FlakyAsset()

        def get_processable_assets(self, _names_filter=None):
            return [self.asset]

    filled = _fill_prediction_gaps(
        scene_path,
        {source: "Steel"},
        FakeManifest(),
    )

    assert filled[source] == "Steel"


def test_fill_prediction_gaps_suffix_guard_handles_short_fake_mesh_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakePath(str):
        pass

    class FakeParent:
        def GetPath(self) -> FakePath:
            return FakePath("/S")

    class FakePrim:
        def IsValid(self) -> bool:
            return True

        def GetTypeName(self) -> str:
            return "Mesh"

        def GetPath(self) -> FakePath:
            return FakePath("/S/M")

        def GetParent(self) -> FakeParent:
            return FakeParent()

    class FakeStage:
        def GetPrimAtPath(self, path: str) -> FakePrim:
            return FakePrim()

    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(lambda *_args: FakeStage()))
    monkeypatch.setattr(Usd, "PrimRange", lambda _root: [FakePrim()])

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="short",
                name="Short",
                prim_path="/S",
                status="completed",
            )
        ]
    )

    filled = _fill_prediction_gaps(
        tmp_path / "scene.usda",
        {"/Long/A/B/C/D/E/F": "Steel"},
        manifest,
    )

    assert "/S/M" not in filled


def test_load_material_library_resolves_relative_paths_and_bindings(
    tmp_path: Path,
) -> None:
    library_dir = tmp_path / "materials"
    library_dir.mkdir()
    library_usd = _create_layer(library_dir / "library.usda")
    yaml_path = tmp_path / "materials.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "library_path: materials/library.usda",
                "entries:",
                "  - name: Steel",
                "    binding: /World/Looks/Steel",
                "  - name: Incomplete",
                "  - binding: /World/Looks/MissingName",
            ]
        ),
        encoding="utf-8",
    )

    resolved_library, name_to_prim = _load_material_library(yaml_path)

    assert resolved_library == library_usd.resolve()
    assert name_to_prim == {"Steel": "/World/Looks/Steel"}


def test_load_material_library_supports_nested_schema_and_prim_path(
    tmp_path: Path,
) -> None:
    library_usd = _create_layer(tmp_path / "library.usda")
    yaml_path = tmp_path / "materials.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "materials:",
                "  library_path: library.usda",
                "  entries:",
                "    - name: Steel",
                "      prim_path: /World/Looks/Steel",
                "    - name: Copper",
                "      binding: /World/Looks/Copper",
                "    - not-a-dict",
            ]
        ),
        encoding="utf-8",
    )

    resolved_library, name_to_prim = _load_material_library(yaml_path)

    assert resolved_library == library_usd.resolve()
    assert name_to_prim == {
        "Steel": "/World/Looks/Steel",
        "Copper": "/World/Looks/Copper",
    }


def test_load_payload_predictions_prefers_explicit_path_and_ignores_invalid_json(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "payload_work"
    _write_jsonl(
        working_dir / "predictions" / "predictions.jsonl",
        [{"id": "/Root/RawMesh", "materials": "Wrong"}],
    )
    explicit = _write_jsonl(
        tmp_path / "payload_predictions.jsonl",
        [
            "",
            {"id": "/Root/MeshA", "materials": {"material": "Steel"}},
            "{broken-json",
            {"id": "/Root/MeshB", "material": "Plastic"},
        ],
    )

    payload = PayloadGroup(
        id="pg",
        group_name="Payload",
        payload_file=str(tmp_path / "payload.usda"),
        working_dir=str(working_dir),
        predictions_path=str(explicit),
        status="completed",
    )

    assert _load_payload_predictions(payload) == {
        "/Root/MeshA": "Steel",
        "/Root/MeshB": "Plastic",
    }

    missing_explicit = PayloadGroup(
        id="pg-raw",
        group_name="PayloadRaw",
        payload_file=str(tmp_path / "payload.usda"),
        working_dir=str(working_dir),
        predictions_path=str(tmp_path / "missing.jsonl"),
        status="completed",
    )
    assert _load_payload_predictions(missing_explicit) == {"/Root/RawMesh": "Wrong"}
    assert (
        _load_payload_predictions(
            PayloadGroup(
                id="pg-empty",
                group_name="PayloadEmpty",
                payload_file=str(tmp_path / "payload.usda"),
                status="completed",
            )
        )
        == {}
    )


def test_build_cascaded_payload_map_rewrites_parent_outputs_bottom_up(
    tmp_path: Path,
) -> None:
    child_orig = _create_layer(tmp_path / "child_orig.usda")
    child_output = _create_layer(
        tmp_path / "child_output.usda",
        sublayers=[str(child_orig.resolve())],
    )
    parent_orig = _create_layer(
        tmp_path / "parent_orig.usda",
        sublayers=[str(child_orig.resolve())],
    )
    parent_output = _create_layer(
        tmp_path / "parent_output.usda",
        sublayers=[str(parent_orig.resolve()), "keep.usda"],
    )
    modified_input = _create_layer(tmp_path / "optimized_input.usda")
    orphan_orig = _create_layer(tmp_path / "orphan_orig.usda")

    manifest = SceneManifest(
        payload_groups=[
            PayloadGroup(
                id="child",
                group_name="child",
                payload_file=str(child_orig),
                output_usd_path=str(child_output),
                depth=0,
                status="completed",
            ),
            PayloadGroup(
                id="parent",
                group_name="parent",
                payload_file=str(parent_orig),
                output_usd_path=str(parent_output),
                depth=1,
                status="completed",
            ),
            PayloadGroup(
                id="no-output",
                group_name="orphan",
                payload_file=str(orphan_orig),
                modified_input_path=str(modified_input),
                depth=2,
                status="completed",
            ),
            PayloadGroup(
                id="duplicate-no-output",
                group_name="duplicate",
                payload_file=str(child_orig),
                modified_input_path=str(modified_input),
                depth=3,
                status="completed",
            ),
        ]
    )

    calls: list[tuple[str, dict[str, str]]] = []

    def fake_rewrite_arcs_in_layer(layer, cascaded_map, resolve_from):
        calls.append((Path(resolve_from).name, dict(cascaded_map)))
        if Path(resolve_from) == child_orig:
            return 0

        child_abs = str(child_orig.resolve())
        layer.subLayerPaths = [cascaded_map[child_abs]]
        return 1

    cascaded = _build_cascaded_payload_map(
        manifest=manifest,
        output_dir=tmp_path / "out",
        rewrite_arcs_in_layer=fake_rewrite_arcs_in_layer,
        shutil=shutil,
    )

    child_abs = str(child_orig.resolve())
    parent_abs = str(parent_orig.resolve())
    orphan_abs = str(orphan_orig.resolve())
    parent_copy = Path(cascaded[parent_abs])
    parent_base = tmp_path / "out" / "payload_copies" / "parent_base.usd"

    assert cascaded[child_abs] == str(child_output)
    assert cascaded[parent_abs] == str(parent_copy)
    assert cascaded[orphan_abs] == str(modified_input)
    assert calls[0] == ("child_orig.usda", {})
    assert calls[1][0] == "parent_orig.usda"
    assert calls[1][1][child_abs] == str(child_output)

    parent_base_layer = Sdf.Layer.FindOrOpen(str(parent_base))
    assert parent_base_layer is not None
    assert parent_base_layer.subLayerPaths == [str(child_output)]

    parent_copy_layer = Sdf.Layer.FindOrOpen(str(parent_copy))
    assert parent_copy_layer is not None
    assert parent_copy_layer.subLayerPaths == [str(parent_base.resolve()), "keep.usda"]


def test_propagate_instance_bindings_falls_back_for_renamed_member_meshes(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/Forklift/node1480")
    UsdGeom.Xform.Define(stage, "/Root/Forklift/node1480/node1479")
    UsdGeom.Mesh.Define(stage, "/Root/Forklift/node1480/node1479/mesh620")
    UsdGeom.Xform.Define(stage, "/Root/Forklift_1/node1483")
    UsdGeom.Xform.Define(stage, "/Root/Forklift_1/node1483/node1482")
    UsdGeom.Mesh.Define(stage, "/Root/Forklift_1/node1483/node1482/mesh432")
    stage.Save()

    output_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="forklift-rep",
                name="node1480",
                prim_path="/Root/Forklift/node1480",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="forklift_node",
                source_file=None,
                instance_count=2,
                member_paths=[
                    "/Root/Forklift/node1480",
                    "/Root/Forklift_1/node1483",
                ],
                representative_id="forklift-rep",
            )
        ],
    )

    written = _propagate_instance_bindings(
        manifest,
        {"/Root/Forklift/node1480/node1479/mesh620": "Car Paint Orange"},
        {"Car Paint Orange": "/Root/Looks/Car_Paint_Orange"},
        output_layer,
        scene_usd_path=scene_path,
    )

    target_spec = output_layer.GetPrimAtPath(
        "/Root/Forklift_1/node1483/node1482/mesh432"
    )
    assert written == 1
    assert target_spec is not None
    assert target_spec.relationships["material:binding"].targetPathList.explicitItems[
        0
    ] == Sdf.Path("/Root/Looks/Car_Paint_Orange")


def test_propagate_instance_bindings_preserves_ordered_member_materials(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/BenchA/node100")
    UsdGeom.Mesh.Define(stage, "/Root/BenchA/node100/mesh10")
    UsdGeom.Xform.Define(stage, "/Root/BenchA/node101")
    UsdGeom.Mesh.Define(stage, "/Root/BenchA/node101/mesh11")
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node200")
    UsdGeom.Mesh.Define(stage, "/Root/BenchB/node200/mesh20")
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node201")
    UsdGeom.Mesh.Define(stage, "/Root/BenchB/node201/mesh21")
    stage.Save()

    output_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="bench-rep",
                name="BenchA",
                prim_path="/Root/BenchA",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="bench",
                source_file=None,
                instance_count=2,
                member_paths=["/Root/BenchA", "/Root/BenchB"],
                representative_id="bench-rep",
            )
        ],
    )

    written = _propagate_instance_bindings(
        manifest,
        {
            "/Root/BenchA/node100/mesh10": "Steel Painted Gray",
            "/Root/BenchA/node101/mesh11": "Aluminum",
        },
        {
            "Steel Painted Gray": "/Root/Looks/Steel_Painted_Gray",
            "Aluminum": "/Root/Looks/Aluminum",
        },
        output_layer,
        scene_usd_path=scene_path,
    )

    top_spec = output_layer.GetPrimAtPath("/Root/BenchB/node200/mesh20")
    frame_spec = output_layer.GetPrimAtPath("/Root/BenchB/node201/mesh21")
    assert written == 2
    assert top_spec is not None
    assert frame_spec is not None
    assert top_spec.relationships["material:binding"].targetPathList.explicitItems[
        0
    ] == Sdf.Path("/Root/Looks/Steel_Painted_Gray")
    assert frame_spec.relationships["material:binding"].targetPathList.explicitItems[
        0
    ] == Sdf.Path("/Root/Looks/Aluminum")


def test_propagate_instance_bindings_uses_ancestor_source_materials(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/BenchA/node100")
    UsdGeom.Mesh.Define(stage, "/Root/BenchA/node100/mesh10")
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node200")
    UsdGeom.Mesh.Define(stage, "/Root/BenchB/node200/mesh20")
    stage.DefinePrim("/Root/BenchB/node200/mesh20/SubsetA", "GeomSubset")
    stage.Save()

    output_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="bench-rep",
                name="BenchA",
                prim_path="/Root/BenchA",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="bench",
                source_file=None,
                instance_count=2,
                member_paths=["/Root/BenchA", "/Root/BenchB"],
                representative_id="bench-rep",
            )
        ],
    )

    written = _propagate_instance_bindings(
        manifest,
        {"/Root/BenchA/node100": "Steel Painted Gray"},
        {"Steel Painted Gray": "/Root/Looks/Steel_Painted_Gray"},
        output_layer,
        scene_usd_path=scene_path,
    )

    target_spec = output_layer.GetPrimAtPath("/Root/BenchB/node200/mesh20")
    assert written == 2
    assert target_spec is not None
    assert target_spec.relationships["material:binding"].targetPathList.explicitItems[
        0
    ] == Sdf.Path("/Root/Looks/Steel_Painted_Gray")
    subset_spec = output_layer.GetPrimAtPath("/Root/BenchB/node200/mesh20/SubsetA")
    assert subset_spec is not None
    assert subset_spec.relationships["material:binding"].targetPathList.explicitItems[
        0
    ] == Sdf.Path("/Root/Looks/Steel_Painted_Gray")


def test_propagate_instance_bindings_falls_back_to_dominant_on_count_mismatch(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/BenchA/node100")
    UsdGeom.Mesh.Define(stage, "/Root/BenchA/node100/mesh10")
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node200")
    UsdGeom.Mesh.Define(stage, "/Root/BenchB/node200/mesh20")
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node201")
    UsdGeom.Mesh.Define(stage, "/Root/BenchB/node201/mesh21")
    stage.Save()

    output_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="bench-rep",
                name="BenchA",
                prim_path="/Root/BenchA",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="bench",
                source_file=None,
                instance_count=2,
                member_paths=["/Root/BenchA", "/Root/BenchB"],
                representative_id="bench-rep",
            )
        ],
    )

    written = _propagate_instance_bindings(
        manifest,
        {"/Root/BenchA/node100/mesh10": "Steel Painted Gray"},
        {"Steel Painted Gray": "/Root/Looks/Steel_Painted_Gray"},
        output_layer,
        scene_usd_path=scene_path,
    )

    assert written == 2
    for path in ["/Root/BenchB/node200/mesh20", "/Root/BenchB/node201/mesh21"]:
        target_spec = output_layer.GetPrimAtPath(path)
        assert target_spec is not None
        assert target_spec.relationships[
            "material:binding"
        ].targetPathList.explicitItems[0] == Sdf.Path("/Root/Looks/Steel_Painted_Gray")


def test_ordered_mesh_binding_falls_back_on_structural_mismatch(
    tmp_path: Path,
    caplog,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/BenchA/node100")
    _define_mesh_with_points(stage, "/Root/BenchA/node100/mesh10", 3)
    UsdGeom.Xform.Define(stage, "/Root/BenchA/node101")
    _define_mesh_with_points(stage, "/Root/BenchA/node101/mesh11", 4)
    UsdGeom.Xform.Define(stage, "/Root/BenchA/node102")
    _define_mesh_with_points(stage, "/Root/BenchA/node102/mesh12", 6)
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node200")
    _define_mesh_with_points(stage, "/Root/BenchB/node200/mesh20", 6)
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node201")
    _define_mesh_with_points(stage, "/Root/BenchB/node201/mesh21", 3)
    UsdGeom.Xform.Define(stage, "/Root/BenchB/node202")
    _define_mesh_with_points(stage, "/Root/BenchB/node202/mesh22", 4)
    stage.Save()

    output_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="bench-rep",
                name="BenchA",
                prim_path="/Root/BenchA",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="bench",
                source_file=None,
                instance_count=2,
                member_paths=["/Root/BenchA", "/Root/BenchB"],
                representative_id="bench-rep",
            )
        ],
    )

    caplog.set_level(logging.WARNING, logger="material_agent.scene.collect")
    written = _propagate_instance_bindings(
        manifest,
        {
            "/Root/BenchA/node100/mesh10": "Steel Painted Gray",
            "/Root/BenchA/node101/mesh11": "Steel Painted Gray",
            "/Root/BenchA/node102/mesh12": "Aluminum",
        },
        {
            "Steel Painted Gray": "/Root/Looks/Steel_Painted_Gray",
            "Aluminum": "/Root/Looks/Aluminum",
        },
        output_layer,
        scene_usd_path=scene_path,
    )

    assert written == 3
    assert "structural fingerprint mismatch" in caplog.text
    for path in [
        "/Root/BenchB/node200/mesh20",
        "/Root/BenchB/node201/mesh21",
        "/Root/BenchB/node202/mesh22",
    ]:
        target_spec = output_layer.GetPrimAtPath(path)
        assert target_spec is not None
        assert target_spec.relationships[
            "material:binding"
        ].targetPathList.explicitItems[0] == Sdf.Path("/Root/Looks/Steel_Painted_Gray")


def test_propagate_instance_bindings_skips_missing_representatives_and_materials(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(stage, "/Root/Member")
    UsdGeom.Mesh.Define(stage, "/Root/Member/Sub/Mesh")
    UsdGeom.Mesh.Define(stage, "/Root/Other/Mesh")
    UsdGeom.Mesh.Define(stage, "/Root/NoSource/Mesh")
    stage.Save()

    output_layer = Sdf.Layer.CreateNew(str(tmp_path / "out.usda"))
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="rep-desc",
                name="DescRep",
                prim_path="/Root/Member/Sub",
                status="completed",
            ),
            SubAsset(
                id="rep-nosource",
                name="NoSource",
                prim_path="/Root/NoSource",
                status="completed",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="no_rep",
                representative_id=None,
                member_paths=["/Root/Member", "/Root/Other"],
            ),
            InstanceGroup(
                group_name="missing_rep",
                representative_id="does-not-exist",
                member_paths=["/Root/Member", "/Root/Other"],
            ),
            InstanceGroup(
                group_name="no_source_bindings",
                representative_id="rep-nosource",
                member_paths=["/Root/NoSource", "/Root/Other"],
            ),
            InstanceGroup(
                group_name="descendant_rep",
                representative_id="rep-desc",
                member_paths=["/Root/Member", "/Root/Other"],
            ),
        ],
    )

    written = _propagate_instance_bindings(
        manifest,
        {
            "/Root/Member/Sub/Mesh": "Ghost",
            "/Root/Other/Mesh": "Ghost",
        },
        {},
        output_layer,
        scene_usd_path=scene_path,
    )

    assert written == 0


def test_path_to_filename_normalizes_prim_paths() -> None:
    assert _path_to_filename("/World/Foo/Bar") == "world_foo_bar"
