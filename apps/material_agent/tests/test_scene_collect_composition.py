# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused composition tests for material_agent.scene.collect."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pxr import Sdf, Tf, Usd, UsdGeom, UsdShade

import material_agent.scene.collect as collect_mod
from material_agent.scene.collect import (
    _compose_prototype_payloads,
    _copy_materials_from_library,
    _create_instance_propagation_layers,
    _create_payload_material_layer,
    _fill_prediction_gaps,
    _process_payload_groups,
    _remap_instance_group_bindings,
    _remove_prim_spec,
    _rewrite_scene_payload_arcs,
    _strip_sublayers,
    _write_binding_over,
    _write_payload_arcs,
    apply_and_compose,
    author_projected_material_layer,
    compose_material_layers,
)
from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)


def _write_jsonl(path: Path, lines: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def _create_library(tmp_path: Path, material_name: str = "Steel") -> tuple[Path, Path]:
    library_dir = tmp_path / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    library_usd = library_dir / "materials.usda"
    stage = Usd.Stage.CreateNew(str(library_usd))
    root = UsdGeom.Scope.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdShade.Material.Define(stage, f"/World/Looks/{material_name}")
    stage.GetRootLayer().Save()

    yaml_path = tmp_path / "materials.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"library_path: {library_usd.parent.name}/{library_usd.name}",
                "entries:",
                f"  - name: {material_name}",
                f"    binding: /World/Looks/{material_name}",
            ]
        ),
        encoding="utf-8",
    )
    return library_usd, yaml_path


def _binding_targets(layer: Sdf.Layer, prim_path: str) -> list[str]:
    spec = layer.GetPrimAtPath(prim_path)
    if not spec:
        return []
    rel = spec.relationships.get("material:binding")
    if not rel:
        return []
    return [str(t) for t in rel.targetPathList.explicitItems]


def _author_binding(layer: Sdf.Layer, prim_path: str, material_path: str) -> None:
    prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
    prim_spec.specifier = Sdf.SpecifierOver
    prim_spec.SetInfo(
        "apiSchemas",
        Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"]),
    )
    rel = Sdf.RelationshipSpec(prim_spec, "material:binding")
    rel.targetPathList.explicitItems = [Sdf.Path(material_path)]


def _make_scene_with_members(scene_path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    for member in ["RepMember", "DupeMember"]:
        UsdGeom.Xform.Define(stage, f"/Root/{member}")
        UsdGeom.Mesh.Define(stage, f"/Root/{member}/Mesh")
    stage.GetRootLayer().Save()

    # Author GeomSubset specs directly on the root layer so binding-over helper
    # sees them when propagating to duplicate members.
    layer = Sdf.Layer.FindOrOpen(str(scene_path))
    for member in ["RepMember", "DupeMember"]:
        subset = Sdf.CreatePrimInLayer(layer, f"/Root/{member}/Mesh/Diffuse_0")
        subset.typeName = "GeomSubset"
    layer.Save()
    return scene_path


def test_apply_and_compose_copies_materials_and_propagates_instance_bindings(
    tmp_path: Path,
) -> None:
    scene_path = _make_scene_with_members(tmp_path / "scene.usda")
    _library_usd, library_yaml = _create_library(tmp_path)

    predictions = _write_jsonl(
        tmp_path / "rep_work" / "predictions" / "predictions.jsonl",
        [{"id": "/Root/RepMember/Mesh", "materials": {"material": "Steel"}}],
    )
    rep = SubAsset(
        id="rep",
        name="Representative",
        prim_path="/Root/RepMember",
        working_dir=str(predictions.parent.parent),
        status="completed",
    )
    dupe = SubAsset(
        id="dupe",
        name="Duplicate",
        prim_path="/Root/DupeMember",
        status="pending",
        instance_group="dup_group",
    )
    manifest = SceneManifest(
        sub_assets=[rep, dupe],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep",
                member_paths=["/Root/RepMember", "/Root/DupeMember"],
            )
        ],
    )

    output_usd = tmp_path / "output" / "composed.usda"
    result = apply_and_compose(scene_path, manifest, output_usd, library_yaml)

    assert result == output_usd
    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer is not None
    assert layer.defaultPrim == "Root"
    assert layer.GetPrimAtPath("/Root/Looks/Steel") is not None
    assert layer.GetPrimAtPath("/Root/Looks").typeName == "Scope"
    assert _binding_targets(layer, "/Root/RepMember/Mesh") == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, "/Root/DupeMember/Mesh") == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, "/Root/DupeMember/Mesh/Diffuse_0") == [
        "/Root/Looks/Steel"
    ]


def test_apply_and_compose_binds_renamed_duplicate_member_paths(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    rep_mesh = "/Root/RepContainer/RepPart/Assembly/PartA/shape/mesh"
    dupe_mesh = "/Root/RenamedContainer/ClonePart/Assembly/PartA/shape/mesh"
    UsdGeom.Mesh.Define(stage, rep_mesh)
    UsdGeom.Mesh.Define(stage, dupe_mesh)
    stage.GetRootLayer().Save()

    _library_usd, library_yaml = _create_library(tmp_path)
    predictions = _write_jsonl(
        tmp_path / "rep_work" / "predictions" / "predictions.jsonl",
        [{"id": rep_mesh, "materials": {"material": "Steel"}}],
    )
    rep = SubAsset(
        id="rep",
        name="Representative",
        prim_path="/Root/RepContainer",
        working_dir=str(predictions.parent.parent),
        status="completed",
    )
    dupe = SubAsset(
        id="dupe",
        name="Duplicate",
        prim_path="/Root/RenamedContainer",
        status="pending",
        instance_group="dup_group",
    )
    manifest = SceneManifest(
        sub_assets=[rep, dupe],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep",
                member_paths=["/Root/RepContainer", "/Root/RenamedContainer"],
            )
        ],
    )

    output_usd = tmp_path / "output" / "composed.usda"
    apply_and_compose(scene_path, manifest, output_usd, library_yaml)

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer is not None
    assert _binding_targets(layer, rep_mesh) == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, dupe_mesh) == ["/Root/Looks/Steel"]
    assert (
        _binding_targets(
            layer,
            "/Root/RenamedContainer/RepPart/Assembly/PartA/shape/mesh",
        )
        == []
    )


def test_apply_and_compose_keeps_representative_fallback_for_partial_member_predictions(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    for member in ["Rep", "Dupe"]:
        UsdGeom.Mesh.Define(stage, f"/Root/{member}/MeshA")
        UsdGeom.Mesh.Define(stage, f"/Root/{member}/MeshB")
    stage.GetRootLayer().Save()

    _library_usd, library_yaml = _create_library(tmp_path)
    predictions = _write_jsonl(
        tmp_path / "rep_work" / "predictions" / "predictions.jsonl",
        [
            {"id": "/Root/Rep/MeshA", "materials": {"material": "Steel"}},
            {"id": "/Root/Rep/MeshB", "materials": {"material": "Steel"}},
            {"id": "/Root/Dupe/MeshA", "materials": {"material": "Steel"}},
            {"id": "/Root/Dupe/MissingMesh", "materials": {"material": "Steel"}},
        ],
    )
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="rep",
                name="Representative",
                prim_path="/Root/Rep",
                working_dir=str(predictions.parent.parent),
                status="completed",
            ),
            SubAsset(
                id="dupe",
                name="Duplicate",
                prim_path="/Root/Dupe",
                status="pending",
                instance_group="dup_group",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep",
                member_paths=["/Root/Rep", "/Root/Dupe"],
            )
        ],
    )

    output_usd = tmp_path / "output" / "composed.usda"
    apply_and_compose(scene_path, manifest, output_usd, library_yaml)

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer is not None
    assert _binding_targets(layer, "/Root/Dupe/MeshA") == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, "/Root/Dupe/MeshB") == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, "/Root/Dupe/MissingMesh") == []


def test_apply_and_compose_suffix_gap_fill_stays_with_selected_members(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    rep_mesh = "/Root/SelectedRep/Source/Assembly/PartA/shape/mesh"
    dupe_mesh = "/Root/SelectedDupe/Renamed/Assembly/PartA/shape/mesh"
    unrelated_mesh = "/Root/Unselected/Renamed/Assembly/PartA/shape/mesh"
    UsdGeom.Mesh.Define(stage, rep_mesh)
    UsdGeom.Mesh.Define(stage, dupe_mesh)
    UsdGeom.Mesh.Define(stage, unrelated_mesh)
    stage.GetRootLayer().Save()

    _library_usd, library_yaml = _create_library(tmp_path)
    predictions = _write_jsonl(
        tmp_path / "rep_work" / "predictions" / "predictions.jsonl",
        [{"id": rep_mesh, "materials": {"material": "Steel"}}],
    )
    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="rep",
                name="Representative",
                prim_path="/Root/SelectedRep",
                working_dir=str(predictions.parent.parent),
                status="completed",
            ),
            SubAsset(
                id="dupe",
                name="Duplicate",
                prim_path="/Root/SelectedDupe",
                status="pending",
                instance_group="dup_group",
            ),
            SubAsset(
                id="unselected",
                name="Unselected",
                prim_path="/Root/Unselected",
                status="completed",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep",
                member_paths=["/Root/SelectedRep", "/Root/SelectedDupe"],
            )
        ],
    )

    output_usd = tmp_path / "output" / "composed.usda"
    apply_and_compose(
        scene_path,
        manifest,
        output_usd,
        library_yaml,
        names_filter=["Representative"],
    )

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer is not None
    assert _binding_targets(layer, dupe_mesh) == ["/Root/Looks/Steel"]
    assert _binding_targets(layer, unrelated_mesh) == []


def test_apply_and_compose_warns_for_unknown_and_uncopyable_materials(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Root/KnownMesh")
    UsdGeom.Mesh.Define(stage, "/Root/UnknownMesh")
    stage.GetRootLayer().Save()

    library_usd = tmp_path / "library.usda"
    library_layer = Sdf.Layer.CreateNew(str(library_usd))
    library_layer.defaultPrim = "World"
    Sdf.CreatePrimInLayer(library_layer, "/World/Looks")
    library_layer.Save()

    library_yaml = tmp_path / "materials.yaml"
    library_yaml.write_text(
        "\n".join(
            [
                "library_path: library.usda",
                "entries:",
                "  - name: Steel",
                "    binding: /World/Looks/MissingSteel",
            ]
        ),
        encoding="utf-8",
    )
    predictions = _write_jsonl(
        tmp_path / "work" / "predictions" / "predictions.jsonl",
        [
            {"id": "/Root/KnownMesh", "materials": "Steel"},
            {"id": "/Root/UnknownMesh", "materials": "NotInLibrary"},
        ],
    )

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="asset",
                name="Asset",
                prim_path="/Root/KnownMesh",
                working_dir=str(predictions.parent.parent),
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="no_rep",
                representative_id=None,
                member_paths=["/Root"],
            ),
            InstanceGroup(
                group_name="descendant",
                representative_id="asset",
                member_paths=["/Root", "/Root/Clone"],
            ),
        ],
    )

    output_usd = tmp_path / "output" / "composed.usda"
    apply_and_compose(scene_path, manifest, output_usd, library_yaml)

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer is not None
    assert _binding_targets(layer, "/Root/KnownMesh") == []
    assert _binding_targets(layer, "/Root/UnknownMesh") == []


def test_author_projected_material_layer_reports_missing_targets_after_load_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    _library_usd, library_yaml = _create_library(tmp_path)

    class MissingPrim:
        def IsValid(self) -> bool:
            return False

    class FakeStage:
        def GetPrimAtPath(self, _path: str) -> MissingPrim:
            return MissingPrim()

        def Load(self, _path: str) -> None:
            raise Tf.ErrorException("cannot load")

    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(lambda *_args: FakeStage()))

    with pytest.raises(ValueError, match="Projected material targets do not exist"):
        author_projected_material_layer(
            scene_path,
            tmp_path / "output" / "materials.usda",
            library_yaml,
            {"/Root/Missing": "Steel"},
        )


def test_apply_and_compose_keeps_original_scene_after_payload_rewrite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scene_path = _make_scene_with_members(tmp_path / "scene.usda")
    _library_usd, library_yaml = _create_library(tmp_path)
    updated_sublayer = tmp_path / "updated_scene.usda"
    Sdf.Layer.CreateNew(str(updated_sublayer)).Save()

    monkeypatch.setattr(
        collect_mod,
        "_rewrite_scene_payload_arcs",
        lambda **_kwargs: [str(updated_sublayer.resolve())],
    )

    manifest = SceneManifest(
        payload_groups=[
            PayloadGroup(
                id="pg",
                group_name="Payload",
                payload_file=str(tmp_path / "payload.usda"),
                instance_paths=["/Root/RepMember"],
                status="completed",
            )
        ]
    )

    output_usd = tmp_path / "output" / "composed.usda"
    apply_and_compose(scene_path, manifest, output_usd, library_yaml)

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert layer is not None
    assert layer.subLayerPaths == [
        str(updated_sublayer.resolve()),
        str(scene_path.resolve()),
    ]


def test_gap_fill_keeps_payloads_unloaded(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_path))
    payload_root = UsdGeom.Xform.Define(payload_stage, "/PayloadRoot")
    payload_stage.SetDefaultPrim(payload_root.GetPrim())
    UsdGeom.Mesh.Define(payload_stage, "/PayloadRoot/MeshA")
    UsdGeom.Mesh.Define(payload_stage, "/PayloadRoot/MeshB")
    payload_stage.GetRootLayer().Save()

    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    payload_asset = UsdGeom.Xform.Define(stage, "/Root/PayloadAsset")
    payload_asset.GetPrim().GetPayloads().AddPayload(str(payload_path), "/PayloadRoot")
    stage.GetRootLayer().Save()

    filled = _fill_prediction_gaps(
        scene_path,
        {"/Root/PayloadAsset/MeshA": "Steel"},
        SceneManifest(
            sub_assets=[
                SubAsset(
                    id="asset",
                    name="Payload Asset",
                    prim_path="/Root/PayloadAsset",
                    status="completed",
                )
            ]
        ),
    )

    assert "/Root/PayloadAsset/MeshB" not in filled


def test_suffix_gap_fill_skips_non_unanimous_suffix_materials(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    steel_mesh = "/Root/RepA/ContainerA/Assembly/PartA/shape/mesh"
    plastic_mesh = "/Root/RepB/ContainerB/Assembly/PartA/shape/mesh"
    dupe_mesh = "/Root/Dupe/ContainerC/Assembly/PartA/shape/mesh"
    UsdGeom.Mesh.Define(stage, steel_mesh)
    UsdGeom.Mesh.Define(stage, plastic_mesh)
    UsdGeom.Mesh.Define(stage, dupe_mesh)
    stage.GetRootLayer().Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="rep_a",
                name="Rep A",
                prim_path="/Root/RepA",
                status="completed",
            ),
            SubAsset(
                id="rep_b",
                name="Rep B",
                prim_path="/Root/RepB",
                status="completed",
            ),
            SubAsset(
                id="dupe",
                name="Duplicate",
                prim_path="/Root/Dupe",
                status="pending",
                instance_group="dup_group",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep_a",
                member_paths=["/Root/RepA", "/Root/Dupe"],
            )
        ],
    )

    filled = _fill_prediction_gaps(
        scene_path,
        {
            steel_mesh: "Steel",
            plastic_mesh: "Plastic",
        },
        manifest,
    )

    assert dupe_mesh not in filled


def test_copy_materials_from_library_remaps_asset_paths(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    library_usd = library_dir / "materials.usda"
    layer = Sdf.Layer.CreateNew(str(library_usd))
    layer.defaultPrim = "World"
    shader_spec = Sdf.CreatePrimInLayer(layer, "/World/Looks/Steel/Shader")
    attr = Sdf.AttributeSpec(shader_spec, "diffuse_texture", Sdf.ValueTypeNames.Asset)
    attr.default = Sdf.AssetPath("textures/base.png")
    arr_attr = Sdf.AttributeSpec(
        shader_spec, "extra_textures", Sdf.ValueTypeNames.AssetArray
    )
    arr_attr.default = Sdf.AssetPathArray(
        [Sdf.AssetPath("textures/a.png"), Sdf.AssetPath("/absolute/keep.png")]
    )
    layer.Save()

    target_layer = Sdf.Layer.CreateAnonymous()
    output_usd = tmp_path / "output" / "composed.usda"
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    remap = _copy_materials_from_library(
        target_layer,
        library_usd,
        {"Steel": "/World/Looks/Steel", "Missing": "/World/Looks/Missing"},
        output_usd,
        scene_default_prim="Root",
    )

    shader = target_layer.GetPrimAtPath("/Root/Looks/Steel/Shader")
    assert shader is not None
    assert target_layer.GetPrimAtPath("/Root/Looks").typeName == "Scope"
    expected_rel = os.path.relpath(
        (library_dir / "textures" / "base.png").resolve(), output_usd.parent.resolve()
    ).replace("\\", "/")
    assert remap == {"/World/Looks/Steel": "/Root/Looks/Steel"}
    assert shader.attributes["diffuse_texture"].default.path == expected_rel
    arr_paths = [p.path for p in shader.attributes["extra_textures"].default]
    assert arr_paths[0].endswith("library/textures/a.png")
    assert arr_paths[1] == "/absolute/keep.png"


def test_process_payload_groups_creates_scoped_layers_and_payload_arcs(
    tmp_path: Path,
) -> None:
    _library_usd, library_yaml = _create_library(tmp_path)
    payload_path = tmp_path / "payload.usda"
    stage = Usd.Stage.CreateNew(str(payload_path))
    root = UsdGeom.Xform.Define(stage, "/Payload")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Payload/Mesh")
    stage.GetRootLayer().Save()

    predictions = _write_jsonl(
        tmp_path / "payload_work" / "predictions" / "predictions.jsonl",
        [{"id": "/Payload/Mesh", "materials": "Steel"}],
    )
    payload_group = PayloadGroup(
        id="payload-1",
        group_name="payload_a",
        payload_file=str(payload_path),
        predictions_path=str(predictions),
        instance_paths=["/Root/InstanceA"],
        status="completed",
    )
    manifest = SceneManifest(payload_groups=[payload_group])

    composed_layer = Sdf.Layer.CreateAnonymous()
    output_usd = tmp_path / "output" / "scene.usda"
    arcs = _process_payload_groups(
        manifest,
        composed_layer,
        output_usd,
        library_yaml,
        {"Steel": "/World/Looks/Steel"},
    )

    assert arcs == 1
    assert payload_group.material_layer_path is not None
    payload_layer = Sdf.Layer.FindOrOpen(payload_group.material_layer_path)
    assert payload_layer is not None
    assert payload_layer.defaultPrim == "Payload"
    assert payload_layer.GetPrimAtPath("/Payload/Looks/Steel") is not None
    assert payload_layer.GetPrimAtPath("/Payload/Looks").typeName == "Scope"
    assert _binding_targets(payload_layer, "/Payload/Mesh") == ["/Payload/Looks/Steel"]
    instance_spec = composed_layer.GetPrimAtPath("/Root/InstanceA")
    assert instance_spec is not None
    assert len(instance_spec.payloadList.prependedItems) == 1
    assert instance_spec.payloadList.prependedItems[0].assetPath.endswith(
        "payload_layers/payload_a.usd"
    )


def test_process_payload_groups_skips_pending_and_predictionless_groups(
    tmp_path: Path,
) -> None:
    _library_usd, library_yaml = _create_library(tmp_path)
    composed_layer = Sdf.Layer.CreateAnonymous()
    manifest = SceneManifest(
        payload_groups=[
            PayloadGroup(
                id="pending",
                group_name="Pending",
                payload_file=str(tmp_path / "pending.usda"),
                status="pending",
            ),
            PayloadGroup(
                id="empty",
                group_name="Empty",
                payload_file=str(tmp_path / "empty.usda"),
                status="completed",
            ),
        ]
    )

    arcs = _process_payload_groups(
        manifest,
        composed_layer,
        tmp_path / "output" / "scene.usda",
        library_yaml,
        {"Steel": "/World/Looks/Steel"},
    )

    assert arcs == 0


def test_create_payload_material_layer_scopes_paths_and_removes_roots(
    tmp_path: Path,
) -> None:
    payload_layer = tmp_path / "payload_layers" / "payload.usda"

    _create_payload_material_layer(
        payload_layer_path=payload_layer,
        default_prim="Payload",
        predictions={
            "/Payload/MeshA": "Scoped",
            "/Payload/MeshB": "World",
            "/Payload/MeshC": "RootOnly",
        },
        used_materials={
            "Scoped": "/Payload/Looks/Scoped",
            "World": "/World/Looks/World",
            "RootOnly": "/Mat",
        },
        library_usd_path=None,
        name_to_prim={
            "Scoped": "/Payload/Looks/Scoped",
            "World": "/World/Looks/World",
            "RootOnly": "/Mat",
            "Unused": "/World/Looks/Unused",
        },
    )
    layer = Sdf.Layer.FindOrOpen(str(payload_layer))
    assert layer is not None
    assert _binding_targets(layer, "/Payload/MeshA") == ["/Payload/Looks/Scoped"]
    assert _binding_targets(layer, "/Payload/MeshB") == ["/Payload/Looks/World"]
    assert _binding_targets(layer, "/Payload/MeshC") == ["/Payload/Mat"]

    plain_layer = tmp_path / "payload_layers" / "plain.usda"
    _create_payload_material_layer(
        payload_layer_path=plain_layer,
        default_prim="",
        predictions={"/Mesh": "Plain"},
        used_materials={"Plain": "/World/Looks/Plain"},
        library_usd_path=None,
        name_to_prim={"Plain": "/World/Looks/Plain"},
    )
    plain = Sdf.Layer.FindOrOpen(str(plain_layer))
    assert plain is not None
    assert _binding_targets(plain, "/Mesh") == ["/World/Looks/Plain"]

    root_layer = Sdf.Layer.CreateAnonymous()
    Sdf.CreatePrimInLayer(root_layer, "/Root")
    _remove_prim_spec(root_layer, "/Missing")
    _remove_prim_spec(root_layer, "/Root")
    assert root_layer.GetPrimAtPath("/Root") is None

    class FakeSpec:
        name = "Root"

    class FakeRoot:
        def __init__(self) -> None:
            self.nameChildren = {"Root": FakeSpec()}

    class FakeLayer:
        def __init__(self) -> None:
            self.pseudoRoot = FakeRoot()

        def GetPrimAtPath(self, path: object) -> object | None:
            return FakeSpec() if str(path) == "/Root" else None

    fake_layer = FakeLayer()
    _remove_prim_spec(fake_layer, "/Root")
    assert fake_layer.pseudoRoot.nameChildren == {}


def test_write_payload_arcs_handles_absolute_fallback_and_create_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layer = Sdf.Layer.CreateAnonymous()
    payload_layer_path = tmp_path / "payload_layers" / "payload.usda"
    payload_layer_path.parent.mkdir()
    payload_layer_path.write_text("#usda 1.0\n")
    output_usd = tmp_path / "output" / "scene.usda"

    monkeypatch.setattr(
        collect_mod.os.path,
        "relpath",
        lambda *_args: (_ for _ in ()).throw(ValueError("different drive")),
    )
    assert (
        _write_payload_arcs(
            layer,
            ["/Root/Instance"],
            payload_layer_path,
            output_usd,
        )
        == 1
    )
    prim_spec = layer.GetPrimAtPath("/Root/Instance")
    assert prim_spec is not None
    assert prim_spec.payloadList.prependedItems[0].assetPath == str(
        payload_layer_path.resolve()
    )

    monkeypatch.setattr(Sdf, "CreatePrimInLayer", lambda *_args: None)
    assert (
        _write_payload_arcs(
            layer,
            ["/Root/Other"],
            payload_layer_path,
            output_usd,
        )
        == 0
    )


def test_compose_prototype_payloads_remaps_bindings_to_prototype_source(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/Root/Prototypes/PayloadProto")
    UsdGeom.Mesh.Define(stage, "/Root/Prototypes/PayloadProto/Mesh")
    stage.OverridePrim("/Root/Instances/PayloadInst")
    stage.GetPrimAtPath(
        "/Root/Instances/PayloadInst"
    ).GetReferences().AddInternalReference(Sdf.Path("/Root/Prototypes/PayloadProto"))
    stage.GetRootLayer().Save()

    manifest = SceneManifest(
        payload_groups=[
            PayloadGroup(
                id="pg",
                group_name="payload_proto",
                payload_file=str(tmp_path / "payload.usda"),
                instance_paths=["/Root/Instances/PayloadInst"],
                status="completed",
            )
        ]
    )
    composed_layer = Sdf.Layer.CreateAnonymous()

    written = _compose_prototype_payloads(
        scene_path,
        manifest,
        {"/Root/Instances/PayloadInst/Mesh": "Steel"},
        {"Steel": "/Root/Looks/Steel"},
        composed_layer,
    )

    assert written == 1
    assert _binding_targets(composed_layer, "/Root/Prototypes/PayloadProto/Mesh") == [
        "/Root/Looks/Steel"
    ]


def test_compose_prototype_payloads_handles_empty_and_missing_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    composed_layer = Sdf.Layer.CreateAnonymous()

    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(lambda *_args: None))
    assert (
        _compose_prototype_payloads(
            tmp_path / "missing.usda",
            SceneManifest(
                payload_groups=[
                    PayloadGroup(
                        id="pg",
                        group_name="Payload",
                        payload_file=str(tmp_path / "payload.usda"),
                        instance_paths=["/Root/Instance"],
                        status="completed",
                    )
                ]
            ),
            {},
            {},
            composed_layer,
        )
        == 0
    )
    monkeypatch.undo()

    scene_path = tmp_path / "prototype-edges.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/Root/Proto")
    ref_no_bindings = stage.OverridePrim("/Root/RefNoBindings")
    ref_no_bindings.GetReferences().AddInternalReference(Sdf.Path("/Root/Proto"))
    ref_missing_material = stage.OverridePrim("/Root/RefMissingMaterial")
    ref_missing_material.GetReferences().AddInternalReference(Sdf.Path("/Root/Proto"))
    stage.GetRootLayer().Save()

    assert (
        _compose_prototype_payloads(
            scene_path,
            SceneManifest(
                payload_groups=[
                    PayloadGroup(
                        id="pending",
                        group_name="Pending",
                        payload_file=str(tmp_path / "payload.usda"),
                        instance_paths=["/Root/RefNoBindings"],
                        status="pending",
                    )
                ]
            ),
            {},
            {},
            composed_layer,
        )
        == 0
    )

    manifest = SceneManifest(
        payload_groups=[
            PayloadGroup(
                id="missing-spec",
                group_name="MissingSpec",
                payload_file=str(tmp_path / "payload.usda"),
                instance_paths=["/Root/Missing"],
                status="completed",
            ),
            PayloadGroup(
                id="no-bindings",
                group_name="NoBindings",
                payload_file=str(tmp_path / "payload.usda"),
                instance_paths=["/Root/RefNoBindings"],
                status="completed",
            ),
            PayloadGroup(
                id="missing-material",
                group_name="MissingMaterial",
                payload_file=str(tmp_path / "payload.usda"),
                instance_paths=["/Root/RefMissingMaterial"],
                status="completed",
            ),
        ]
    )

    assert (
        _compose_prototype_payloads(
            scene_path,
            manifest,
            {"/Root/RefMissingMaterial/Mesh": "Ghost"},
            {},
            composed_layer,
        )
        == 0
    )


def test_remap_instance_group_bindings_direct_no_reference_branch(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Root/Rep/Mesh")
    UsdGeom.Mesh.Define(stage, "/Root/Member/Mesh")
    UsdGeom.Mesh.Define(stage, "/Root/Member/Direct")
    UsdGeom.Mesh.Define(stage, "/Root/Member/Existing")
    stage.GetRootLayer().Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="rep",
                name="Representative",
                prim_path="/Root/Rep",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="direct_group",
                representative_id="rep",
                member_paths=["/Root/Rep", "/Root/Member"],
            )
        ],
    )
    composed_layer = Sdf.Layer.CreateAnonymous()
    _write_binding_over(composed_layer, "/Root/Member/Existing", "/Root/Looks/Steel")

    written = _remap_instance_group_bindings(
        scene_path,
        manifest,
        {
            "/Root/Rep/Mesh": "Steel",
            "/Root/Rep/Ghost": "Ghost",
            "/Root/Member/Direct": "Steel",
            "/Root/Member/MissingMaterial": "Ghost",
            "/Root/Member/MissingTarget": "Steel",
            "/Root/Member/Existing": "Steel",
        },
        {"Steel": "/Root/Looks/Steel"},
        composed_layer,
    )

    assert written == 2
    assert _binding_targets(composed_layer, "/Root/Member/Mesh") == [
        "/Root/Looks/Steel"
    ]
    assert _binding_targets(composed_layer, "/Root/Member/Direct") == [
        "/Root/Looks/Steel"
    ]


def test_remap_instance_group_bindings_prototype_source_and_members(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    for proto_path, mesh_name in [
        ("/Root/Prototypes/RepProto", "Mesh"),
        ("/Root/Prototypes/MemberProto", "Mesh"),
        ("/Root/Prototypes/RenamedProto", "DifferentMesh"),
    ]:
        UsdGeom.Xform.Define(stage, proto_path)
        UsdGeom.Mesh.Define(stage, f"{proto_path}/{mesh_name}")
        UsdGeom.Mesh.Define(stage, f"{proto_path}/Already")

    rep = stage.OverridePrim("/Root/Instances/Rep")
    rep.GetReferences().AddInternalReference(Sdf.Path("/Root/Prototypes/RepProto"))
    member = stage.OverridePrim("/Root/Instances/Member")
    member.GetReferences().AddInternalReference(
        Sdf.Path("/Root/Prototypes/MemberProto")
    )
    same = stage.OverridePrim("/Root/Instances/Same")
    same.GetReferences().AddInternalReference(Sdf.Path("/Root/Prototypes/RepProto"))
    renamed = stage.OverridePrim("/Root/Instances/Renamed")
    renamed.GetReferences().AddInternalReference(
        Sdf.Path("/Root/Prototypes/RenamedProto")
    )
    stage.OverridePrim("/Root/Instances/NoRef")
    stage.GetRootLayer().Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="rep",
                name="Representative",
                prim_path="/Root/Instances/Rep",
                status="completed",
            )
        ],
        instance_groups=[
            InstanceGroup(
                group_name="proto_group",
                representative_id="rep",
                member_paths=[
                    "/Root/Instances/Rep",
                    "/Root/Instances/NoSpec",
                    "/Root/Instances/NoRef",
                    "/Root/Instances/Same",
                    "/Root/Instances/Member",
                    "/Root/Instances/Renamed",
                ],
            )
        ],
    )
    composed_layer = Sdf.Layer.CreateAnonymous()
    _write_binding_over(
        composed_layer,
        "/Root/Prototypes/RepProto/Already",
        "/Root/Looks/Steel",
    )
    _write_binding_over(
        composed_layer,
        "/Root/Prototypes/MemberProto/Already",
        "/Root/Looks/Steel",
    )
    _write_binding_over(
        composed_layer,
        "/Root/Prototypes/RenamedProto/Already",
        "/Root/Looks/Steel",
    )
    fallback_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        collect_mod,
        "_collect_mesh_paths_from_stage",
        lambda _stage, prefix: [f"{prefix}/Mesh"],
    )
    monkeypatch.setattr(
        collect_mod,
        "_collect_mesh_paths_from_layer",
        lambda _layer, prefix: [f"{prefix}/DifferentMesh"],
    )
    monkeypatch.setattr(
        collect_mod,
        "_write_ordered_mesh_bindings",
        lambda **kwargs: fallback_calls.append(kwargs) or 1,
    )

    written = _remap_instance_group_bindings(
        scene_path,
        manifest,
        {
            "/Root/Instances/Rep/Mesh": "Steel",
            "/Root/Instances/Rep/Already": "Steel",
            "/Root/Instances/Rep/Ghost": "Ghost",
            "/Root/Instances/Rep/NoTarget": "Steel",
        },
        {"Steel": "/Root/Looks/Steel"},
        composed_layer,
        skip_paths={"/Root/Instances/Rep"},
    )

    assert written >= 3
    assert _binding_targets(composed_layer, "/Root/Prototypes/RepProto/Mesh") == [
        "/Root/Looks/Steel"
    ]
    assert _binding_targets(composed_layer, "/Root/Prototypes/MemberProto/Mesh") == [
        "/Root/Looks/Steel"
    ]
    assert fallback_calls
    assert fallback_calls[-1]["target_meshes"] == [
        "/Root/Prototypes/RenamedProto/DifferentMesh"
    ]


def test_remap_instance_group_bindings_skips_and_falls_back_for_direct_members(
    monkeypatch,
    tmp_path: Path,
) -> None:
    composed_layer = Sdf.Layer.CreateAnonymous()
    monkeypatch.setattr(Usd.Stage, "Open", staticmethod(lambda *_args: None))
    assert (
        _remap_instance_group_bindings(
            tmp_path / "missing.usda",
            SceneManifest(),
            {},
            {},
            composed_layer,
        )
        == 0
    )
    monkeypatch.undo()

    scene_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Mesh.Define(stage, "/Root/Rep/MeshA")
    UsdGeom.Mesh.Define(stage, "/Root/Member/RenamedMesh")
    UsdGeom.Mesh.Define(stage, "/Root/Desc/Sub/Mesh")
    stage.GetRootLayer().Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="pending-rep",
                name="PendingRep",
                prim_path="/Root/Pending",
                status="pending",
            ),
            SubAsset(
                id="direct-rep",
                name="DirectRep",
                prim_path="/Root/Rep",
                status="completed",
            ),
            SubAsset(
                id="desc-rep",
                name="DescRep",
                prim_path="/Root/Desc/Sub",
                status="completed",
            ),
            SubAsset(
                id="missing-spec",
                name="MissingSpec",
                prim_path="/Root/MissingSpec",
                status="completed",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="no_rep",
                representative_id=None,
                member_paths=["/Root/Rep", "/Root/Member"],
            ),
            InstanceGroup(
                group_name="pending",
                representative_id="pending-rep",
                member_paths=["/Root/Pending", "/Root/Member"],
            ),
            InstanceGroup(
                group_name="descendant",
                representative_id="desc-rep",
                member_paths=["/Root/Desc", "/Root/Member"],
            ),
            InstanceGroup(
                group_name="missing_spec",
                representative_id="missing-spec",
                member_paths=["/Root/MissingSpec", "/Root/Member"],
            ),
            InstanceGroup(
                group_name="direct_fallback",
                representative_id="direct-rep",
                member_paths=["/Root/Rep", "/Root/Member"],
            ),
        ],
    )

    written = _remap_instance_group_bindings(
        scene_path,
        manifest,
        {"/Root/Rep/MeshA": "Steel", "/Root/Desc/Sub/Mesh": "Ghost"},
        {"Steel": "/Root/Looks/Steel"},
        composed_layer,
    )

    assert written == 1
    assert _binding_targets(composed_layer, "/Root/Member/RenamedMesh") == [
        "/Root/Looks/Steel"
    ]


def test_rewrite_scene_payload_arcs_copies_sublayers(tmp_path: Path) -> None:
    sublayer = tmp_path / "scene_sub.usda"
    Sdf.Layer.CreateNew(str(sublayer)).Save()
    scene_path = tmp_path / "scene.usda"
    layer = Sdf.Layer.CreateNew(str(scene_path))
    layer.subLayerPaths = [os.path.relpath(sublayer, scene_path.parent)]
    layer.Save()

    with (
        patch(
            "material_agent.scene.collect._build_cascaded_payload_map",
            return_value={
                str((tmp_path / "payload.usda").resolve()): str(
                    (tmp_path / "updated.usda").resolve()
                )
            },
        ),
        patch(
            "material_agent.scene.payload_dag_utils.rewrite_arcs_in_layer",
            return_value=1,
        ) as mock_rewrite,
    ):
        result = _rewrite_scene_payload_arcs(
            scene_path,
            SceneManifest(),
            tmp_path / "output",
        )

    assert len(result) == 1
    assert Path(result[0]).exists()
    assert Path(result[0]).parent.name == "scene_layers"
    mock_rewrite.assert_called_once()
    assert Path(mock_rewrite.call_args.kwargs["resolve_from"]) == sublayer.resolve()


def test_rewrite_scene_payload_arcs_handles_empty_map_and_missing_sublayers(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.usda"
    layer = Sdf.Layer.CreateNew(str(scene_path))
    layer.subLayerPaths = ["missing_sublayer.usda"]
    layer.Save()

    with patch(
        "material_agent.scene.collect._build_cascaded_payload_map",
        return_value={},
    ):
        assert _rewrite_scene_payload_arcs(
            scene_path,
            SceneManifest(),
            tmp_path / "empty-output",
        ) == [str(scene_path.resolve())]

    with (
        patch(
            "material_agent.scene.collect._build_cascaded_payload_map",
            return_value={
                str((tmp_path / "payload.usda").resolve()): str(
                    (tmp_path / "updated.usda").resolve()
                )
            },
        ),
        patch(
            "material_agent.scene.payload_dag_utils.rewrite_arcs_in_layer",
            return_value=0,
        ),
    ):
        result = _rewrite_scene_payload_arcs(
            scene_path,
            SceneManifest(),
            tmp_path / "missing-sublayer-output",
        )

    assert result == ["missing_sublayer.usda"]


def test_compose_material_layers_strips_sublayers_and_propagates_instances(
    tmp_path: Path,
) -> None:
    scene_path = _make_scene_with_members(tmp_path / "scene.usda")
    rep_layer_path = tmp_path / "rep_output.usda"
    rep_layer = Sdf.Layer.CreateNew(str(rep_layer_path))
    rep_layer.defaultPrim = "Root"
    rep_layer.subLayerPaths = [str((tmp_path / "rep_geometry.usda").resolve())]
    Sdf.CreatePrimInLayer(rep_layer, "/Root/Looks/Steel").specifier = Sdf.SpecifierDef
    _author_binding(rep_layer, "/Root/RepMember/Mesh", "/Root/Looks/Steel")
    rep_layer.Save()
    Sdf.Layer.CreateNew(str(tmp_path / "rep_geometry.usda")).Save()

    rep = SubAsset(
        id="rep",
        name="Representative",
        prim_path="/Root/RepMember",
        material_layer_path=str(rep_layer_path),
        status="completed",
    )
    dupe = SubAsset(
        id="dupe",
        name="Duplicate",
        prim_path="/Root/DupeMember",
        status="pending",
    )
    manifest = SceneManifest(
        sub_assets=[rep, dupe],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id="rep",
                member_paths=["/Root/RepMember", "/Root/DupeMember"],
            )
        ],
    )

    output_usd = tmp_path / "composed" / "scene_composed.usda"
    result = compose_material_layers(scene_path, manifest, output_usd)

    assert result == output_usd
    output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert output_layer is not None
    assert len(output_layer.subLayerPaths) == 3
    assert output_layer.subLayerPaths[-1] == str(scene_path.resolve())

    stripped_layer = Path(output_layer.subLayerPaths[0])
    propagation_layer = Path(output_layer.subLayerPaths[1])
    assert stripped_layer.exists()
    assert propagation_layer.exists()

    stripped = Sdf.Layer.FindOrOpen(str(stripped_layer))
    propagated = Sdf.Layer.FindOrOpen(str(propagation_layer))
    assert stripped is not None and not stripped.subLayerPaths
    assert propagated is not None
    assert _binding_targets(propagated, "/Root/DupeMember/Mesh") == [
        "/Root/Looks/Steel"
    ]


def test_compose_material_layers_skips_missing_duplicate_and_empty_layers(
    tmp_path: Path,
) -> None:
    scene_path = _make_scene_with_members(tmp_path / "scene.usda")
    good_layer_path = tmp_path / "good_output.usda"
    good_layer = Sdf.Layer.CreateNew(str(good_layer_path))
    _author_binding(good_layer, "/Root/RepMember/Mesh", "/Root/Looks/Steel")
    good_layer.Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="pending",
                name="Pending",
                prim_path="/Root/Pending",
                status="pending",
            ),
            SubAsset(
                id="missing",
                name="Missing",
                prim_path="/Root/Missing",
                material_layer_path=str(tmp_path / "missing.usda"),
                status="completed",
            ),
            SubAsset(
                id="good",
                name="Good",
                prim_path="/Root/RepMember",
                material_layer_path=str(good_layer_path),
                status="completed",
            ),
            SubAsset(
                id="duplicate",
                name="Duplicate",
                prim_path="/Root/DupeMember",
                material_layer_path=str(good_layer_path),
                status="completed",
            ),
        ]
    )

    output_usd = tmp_path / "composed" / "scene.usda"
    compose_material_layers(scene_path, manifest, output_usd)

    output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
    assert output_layer is not None
    assert output_layer.subLayerPaths == [
        str(good_layer_path.resolve()),
        str(scene_path.resolve()),
    ]
    assert _strip_sublayers(good_layer_path, tmp_path / "stripped", "good") == (
        good_layer_path.resolve()
    )

    empty_output = tmp_path / "composed" / "empty.usda"
    compose_material_layers(scene_path, SceneManifest(), empty_output)
    empty_layer = Sdf.Layer.FindOrOpen(str(empty_output))
    assert empty_layer is not None
    assert empty_layer.subLayerPaths == [str(scene_path.resolve())]


def test_create_instance_propagation_layers_handles_skips_and_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rep_layer_path = tmp_path / "rep.usda"
    rep_layer = Sdf.Layer.CreateNew(str(rep_layer_path))
    _author_binding(rep_layer, "/Root/Rep/Mesh", "/Root/Looks/Steel")
    rep_layer.Save()
    member_layer_path = tmp_path / "member.usda"
    Sdf.Layer.CreateNew(str(member_layer_path)).Save()

    manifest = SceneManifest(
        sub_assets=[
            SubAsset(
                id="no-layer",
                name="NoLayer",
                prim_path="/Root/NoLayer",
                status="completed",
            ),
            SubAsset(
                id="missing-layer",
                name="MissingLayer",
                prim_path="/Root/MissingLayer",
                material_layer_path=str(tmp_path / "missing.usda"),
                status="completed",
            ),
            SubAsset(
                id="rep",
                name="Rep",
                prim_path="/Root/Rep",
                material_layer_path=str(rep_layer_path),
                status="completed",
            ),
            SubAsset(
                id="member-own",
                name="MemberOwn",
                prim_path="/Root/MemberOwn",
                material_layer_path=str(member_layer_path),
                status="completed",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="no_rep",
                representative_id=None,
                member_paths=["/Root/Rep", "/Root/Member"],
            ),
            InstanceGroup(
                group_name="no_layer",
                representative_id="no-layer",
                member_paths=["/Root/NoLayer", "/Root/Member"],
            ),
            InstanceGroup(
                group_name="missing_layer",
                representative_id="missing-layer",
                member_paths=["/Root/MissingLayer", "/Root/Member"],
            ),
            InstanceGroup(
                group_name="member_own",
                representative_id="rep",
                member_paths=["/Root/Rep", "/Root/MemberOwn"],
            ),
        ],
    )

    assert _create_instance_propagation_layers(manifest, tmp_path / "out") == []

    original_find_or_open = Sdf.Layer.FindOrOpen
    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", staticmethod(lambda _path: None))
    assert (
        _create_instance_propagation_layers(
            SceneManifest(
                sub_assets=[
                    SubAsset(
                        id="rep",
                        name="Rep",
                        prim_path="/Root/Rep",
                        material_layer_path=str(rep_layer_path),
                        status="completed",
                    )
                ],
                instance_groups=[
                    InstanceGroup(
                        group_name="open_none",
                        representative_id="rep",
                        member_paths=["/Root/Rep", "/Root/Member"],
                    )
                ],
            ),
            tmp_path / "out-open-none",
        )
        == []
    )
    monkeypatch.setattr(Sdf.Layer, "FindOrOpen", original_find_or_open)

    monkeypatch.setattr(
        collect_mod,
        "_remap_layer_prims",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        _create_instance_propagation_layers(
            SceneManifest(
                sub_assets=[
                    SubAsset(
                        id="rep",
                        name="Rep",
                        prim_path="/Root/Rep",
                        material_layer_path=str(rep_layer_path),
                        status="completed",
                    )
                ],
                instance_groups=[
                    InstanceGroup(
                        group_name="raises",
                        representative_id="rep",
                        member_paths=["/Root/Rep", "/Root/Member"],
                    )
                ],
            ),
            tmp_path / "out-raises",
        )
        == []
    )
