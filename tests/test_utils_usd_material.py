# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD material asset discovery helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from PIL import Image

pxr = pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from world_understanding.utils.usd import material as material_utils  # noqa: E402
from world_understanding.utils.usd.material import (  # noqa: E402
    add_ovrtx_preview_fallbacks_for_materialx_openpbr,
    add_ovrtx_preview_fallbacks_for_texture_file_materials,
    add_ovrtx_preview_fallbacks_to_stage_file,
    bake_texture_file_materials_to_display_color_for_render,
    convert_custom_mdl_to_builtin,
    ensure_looks_scope,
    ensure_looks_scope_spec,
    get_local_mdl_assets,
    get_local_texture_file_assets,
    localize_package_texture_assets_for_render,
    write_ovrtx_preview_fallback_overlay_for_materialx_openpbr,
)


def test_convert_custom_mdl_to_builtin_restores_blank_omnipbr_source_asset(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    shader = UsdShade.Shader.Define(stage, "/Looks/Metal/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    )
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier",
        Sdf.ValueTypeNames.Token,
    ).Set("OmniPBR")

    convert_custom_mdl_to_builtin(stage)

    mdl_attr = shader.GetPrim().GetAttribute("info:mdl:sourceAsset")
    assert mdl_attr.Get() == Sdf.AssetPath("OmniPBR.mdl")


def test_asset_discovery_prefers_usd_resolved_path_for_sublayer_assets(
    tmp_path: Path,
) -> None:
    asset_dir = tmp_path / "asset"
    texture_dir = asset_dir / "textures"
    material_dir = asset_dir / "materials"
    texture_dir.mkdir(parents=True)
    material_dir.mkdir()
    texture_path = texture_dir / "diffuse.png"
    mdl_path = material_dir / "surface.mdl"
    texture_path.write_bytes(b"not-a-real-png")
    mdl_path.write_text("// test mdl\n", encoding="utf-8")

    sublayer_path = asset_dir / "model.usda"
    sublayer_stage = Usd.Stage.CreateNew(str(sublayer_path))
    texture_shader = UsdShade.Shader.Define(sublayer_stage, "/TextureShader")
    texture_shader.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("textures/diffuse.png"))
    mdl_shader = UsdShade.Shader.Define(sublayer_stage, "/MdlShader")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("materials/surface.mdl"))
    sublayer_stage.Save()

    root_path = tmp_path / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append("asset/model.usda")
    root_stage.Save()
    composed_stage = Usd.Stage.Open(str(root_path))
    assert composed_stage is not None

    texture_assets = get_local_texture_file_assets(
        composed_stage,
        base_dir=tmp_path,
    )
    mdl_assets = get_local_mdl_assets(composed_stage, base_dir=tmp_path)

    assert [asset["resolved_path"] for asset in texture_assets] == [
        str(texture_path.resolve())
    ]
    assert [asset["resolved_path"] for asset in mdl_assets] == [str(mdl_path.resolve())]


def test_texture_asset_discovery_skips_embedded_data_uris(tmp_path: Path) -> None:
    stage_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    texture_shader = UsdShade.Shader.Define(stage, "/TextureShader")
    data_uri = "data:image/png;base64," + ("A" * 600)
    texture_shader.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(data_uri))
    stage.Save()

    texture_assets = get_local_texture_file_assets(stage, base_dir=tmp_path)

    assert texture_assets == [
        {
            "prim_path": "/TextureShader",
            "attr_name": "inputs:file",
            "file_path": data_uri,
            "resolved_path": None,
            "is_local": False,
        }
    ]


def _create_materialx_openpbr_material(
    stage: Usd.Stage,
    material_path: str = "/World/Looks/Gold",
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()
    prim.CreateAttribute("inputs:base_color", Sdf.ValueTypeNames.Color3f).Set(
        (1.0, 0.766, 0.336),
    )
    prim.CreateAttribute("inputs:base_metalness", Sdf.ValueTypeNames.Float).Set(1.0)
    prim.CreateAttribute("inputs:specular_roughness", Sdf.ValueTypeNames.Float).Set(
        0.05,
    )
    prim.CreateAttribute("inputs:geometry_opacity", Sdf.ValueTypeNames.Float).Set(0.8)

    shader = UsdShade.Shader.Define(
        stage,
        f"{material_path}/open_pbr_surface_surfaceshader",
    )
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mtlx").ConnectToSource(shader_output)
    material.CreateSurfaceOutput()
    return material


def _connect_existing_preview_surface(material: UsdShade.Material) -> None:
    stage = material.GetPrim().GetStage()
    material_path = str(material.GetPath())
    shader = UsdShade.Shader.Define(stage, f"{material_path}/ExistingPreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)


def _connected_surface_shader_id(material: UsdShade.Material) -> str | None:
    output = material.GetSurfaceOutput()
    if not output:
        return None
    sources, _ = output.GetConnectedSources()
    if not sources:
        return None
    shader = UsdShade.Shader(sources[0].source.GetPrim())
    return shader.GetIdAttr().Get()


def test_adds_ovrtx_preview_fallback_for_materialx_openpbr() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 1

    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert material.GetSurfaceOutput("mtlx").HasConnectedSource()

    shader = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Gold/OVRTXPreviewSurface"),
    )
    assert shader.GetInput("diffuseColor").Get() == (1.0, 0.766, 0.336)
    assert shader.GetInput("metallic").Get() == 1.0
    assert shader.GetInput("roughness").Get() == pytest.approx(0.05)
    assert shader.GetInput("opacity").Get() == pytest.approx(0.8)


def test_ovrtx_preview_fallback_can_target_material_paths() -> None:
    stage = Usd.Stage.CreateInMemory()
    gold = _create_materialx_openpbr_material(stage, "/World/Looks/Gold")
    silver = _create_materialx_openpbr_material(stage, "/World/Looks/Silver")

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
            target_material_paths=["/World/Looks/Gold"],
        )
        == 1
    )

    assert _connected_surface_shader_id(gold) == "UsdPreviewSurface"
    assert not gold.GetSurfaceOutput("mtlx").HasConnectedSource()
    assert _connected_surface_shader_id(silver) is None
    assert silver.GetSurfaceOutput("mtlx").HasConnectedSource()
    assert not stage.GetPrimAtPath("/World/Looks/Silver/OVRTXPreviewSurface")


def test_ovrtx_preview_fallback_approximates_openpbr_transmission() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    prim = material.GetPrim()
    prim.GetAttribute("inputs:base_color").Set((0.38, 0.38, 0.38))
    prim.GetAttribute("inputs:geometry_opacity").Set(1.0)
    prim.CreateAttribute("inputs:transmission_weight", Sdf.ValueTypeNames.Float).Set(
        1.0,
    )
    prim.CreateAttribute(
        "inputs:transmission_color",
        Sdf.ValueTypeNames.Color3f,
    ).Set((0.82, 0.9, 1.0))

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 1

    shader = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Gold/OVRTXPreviewSurface"),
    )
    assert shader.GetInput("diffuseColor").Get() == (0.82, 0.9, 1.0)
    assert shader.GetInput("opacity").Get() == pytest.approx(0.35)


def test_ovrtx_preview_fallback_ignores_low_inherited_transmission() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    prim = material.GetPrim()
    prim.GetAttribute("inputs:base_color").Set((0.005, 0.005, 0.005))
    prim.GetAttribute("inputs:geometry_opacity").Set(1.0)
    prim.CreateAttribute("inputs:transmission_weight", Sdf.ValueTypeNames.Float).Set(
        0.1,
    )
    prim.CreateAttribute(
        "inputs:transmission_color",
        Sdf.ValueTypeNames.Color3f,
    ).Set((1.0, 1.0, 1.0))

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 1

    shader = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Gold/OVRTXPreviewSurface"),
    )
    assert shader.GetInput("diffuseColor").Get() == (0.005, 0.005, 0.005)
    assert shader.GetInput("opacity").Get() == pytest.approx(1.0)


def test_ovrtx_preview_fallback_is_idempotent_when_surface_exists() -> None:
    stage = Usd.Stage.CreateInMemory()
    _create_materialx_openpbr_material(stage)

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 1
    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 0


def test_ovrtx_preview_fallback_disables_instanceable_material() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    material.GetPrim().SetInstanceable(True)

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert not material.GetPrim().IsInstanceable()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_disables_referenced_material_instance(
    tmp_path: Path,
) -> None:
    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    _create_materialx_openpbr_material(prototype_stage, "/Prototype/Gold")
    prototype_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
    material.GetPrim().GetReferences().AddReference(
        str(prototype_path),
        "/Prototype/Gold",
    )
    material.GetPrim().SetInstanceable(True)
    assert material.GetPrim().IsInstance()

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert not material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_suppresses_materialx_when_surface_exists() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    _connect_existing_preview_surface(material)

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()
    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 0
    )


def test_ovrtx_preview_suppression_disables_instanceable_material() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    _connect_existing_preview_surface(material)
    material.GetPrim().SetInstanceable(True)

    assert (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    assert not material.GetPrim().IsInstanceable()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_preserves_mdl_materials() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = _create_materialx_openpbr_material(stage)
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Gold/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)

    assert add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage) == 0
    assert _connected_surface_shader_id(material) is None


def test_adds_textured_ovrtx_preview_fallback_for_texture_file_material(
    tmp_path: Path,
) -> None:
    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (4, 4), (32, 128, 240)).save(texture_path)

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    material.GetPrim().CreateAttribute(
        "inputs:base_metalness",
        Sdf.ValueTypeNames.Float,
    ).Set(0.25)
    material.GetPrim().CreateAttribute(
        "inputs:specular_roughness",
        Sdf.ValueTypeNames.Float,
    ).Set(0.7)

    assert add_ovrtx_preview_fallbacks_for_texture_file_materials(stage) == 1

    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    preview = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Painted/OVRTXPreviewSurface"),
    )
    assert preview.GetInput("metallic").Get() == pytest.approx(0.25)
    assert preview.GetInput("roughness").Get() == pytest.approx(0.7)
    sources, _ = preview.GetInput("diffuseColor").GetConnectedSources()
    assert sources == []
    assert preview.GetInput("diffuseColor").Get() == pytest.approx(
        (32 / 255, 128 / 255, 240 / 255),
    )
    assert not stage.GetPrimAtPath(
        "/World/Looks/Painted/OVRTXPreviewAlbedoTexture",
    ).IsValid()
    assert add_ovrtx_preview_fallbacks_for_texture_file_materials(stage) == 0


def test_textured_ovrtx_preview_fallback_leaves_mdl_output_connected() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./textures/painted_albedo.png"))
    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Mdl")
    mdl_shader.CreateIdAttr("mdl:OmniPBR")
    mdl_output = mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(mdl_output)

    assert add_ovrtx_preview_fallbacks_for_texture_file_materials(stage) == 1

    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert material.GetSurfaceOutput("mdl").HasConnectedSource()


def test_textured_ovrtx_preview_fallback_skips_material_with_surface() -> None:
    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./textures/painted_albedo.png"))
    _connect_existing_preview_surface(material)

    assert add_ovrtx_preview_fallbacks_for_texture_file_materials(stage) == 0
    assert not stage.GetPrimAtPath(
        "/World/Looks/Painted/OVRTXPreviewAlbedoTexture",
    ).IsValid()


def test_textured_ovrtx_preview_fallback_can_override_existing_surface(
    tmp_path: Path,
) -> None:
    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (4, 4), (5, 150, 20)).save(texture_path)

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    preview = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Painted/ExistingPreviewSurface",
    )
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    albedo = UsdShade.Shader.Define(stage, "/World/Looks/Painted/AlbedoTexture")
    albedo.CreateIdAttr("UsdUVTexture")
    albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path))
    )
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )

    assert add_ovrtx_preview_fallbacks_for_texture_file_materials(stage) == 0
    assert (
        add_ovrtx_preview_fallbacks_for_texture_file_materials(
            stage,
            override_existing_surface=True,
        )
        == 1
    )

    sources, _ = material.GetSurfaceOutput().GetConnectedSources()
    assert sources[0].source.GetPrim().GetName() == "OVRTXPreviewSurface"
    fallback = UsdShade.Shader(sources[0].source.GetPrim())
    fallback_sources, _ = fallback.GetInput("diffuseColor").GetConnectedSources()
    assert fallback_sources == []
    assert fallback.GetInput("diffuseColor").Get() == pytest.approx(
        (5 / 255, 150 / 255, 20 / 255),
    )
    assert (
        add_ovrtx_preview_fallbacks_for_texture_file_materials(
            stage,
            override_existing_surface=True,
        )
        == 0
    )


def test_textured_ovrtx_preview_fallback_can_author_uv_texture_graph(
    tmp_path: Path,
) -> None:
    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (4, 4), (5, 150, 20)).save(texture_path)

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))

    assert (
        add_ovrtx_preview_fallbacks_for_texture_file_materials(
            stage,
            connect_diffuse_texture=True,
        )
        == 1
    )

    preview = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Painted/OVRTXPreviewSurface"),
    )
    sources, _ = preview.GetInput("diffuseColor").GetConnectedSources()
    texture = UsdShade.Shader(sources[0].source.GetPrim())
    assert texture.GetPrim().GetName() == "OVRTXPreviewAlbedoTexture"
    assert texture.GetIdAttr().Get() == "UsdUVTexture"
    assert texture.GetInput("file").Get().path == str(texture_path)
    assert texture.GetInput("sourceColorSpace").Get() == "sRGB"
    assert list(texture.GetInput("fallback").Get()) == pytest.approx(
        [5 / 255, 150 / 255, 20 / 255, 1.0],
    )

    st_sources, _ = texture.GetInput("st").GetConnectedSources()
    reader = UsdShade.Shader(st_sources[0].source.GetPrim())
    assert reader.GetPrim().GetName() == "OVRTXPreviewSTReader"
    assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    assert reader.GetInput("varname").Get() == "st"


def test_textured_ovrtx_preview_fallback_prefers_generated_material_texture(
    tmp_path: Path,
) -> None:
    generated_texture = tmp_path / "generated_albedo.png"
    original_texture = tmp_path / "original_albedo.png"
    Image.new("RGB", (4, 4), (210, 160, 20)).save(generated_texture)
    Image.new("RGB", (4, 4), (20, 60, 210)).save(original_texture)

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(generated_texture)))

    preview = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    original_albedo = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Painted/OriginalAlbedoTexture",
    )
    original_albedo.CreateIdAttr("UsdUVTexture")
    original_albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(original_texture))
    )
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        original_albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )

    assert (
        add_ovrtx_preview_fallbacks_for_texture_file_materials(
            stage,
            override_existing_surface=True,
            connect_diffuse_texture=True,
        )
        == 1
    )

    fallback = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Painted/OVRTXPreviewSurface"),
    )
    sources, _ = fallback.GetInput("diffuseColor").GetConnectedSources()
    texture = UsdShade.Shader(sources[0].source.GetPrim())
    assert texture.GetInput("file").Get().path == str(generated_texture)


def test_textured_ovrtx_preview_fallback_can_read_display_color_primvar(
    tmp_path: Path,
) -> None:
    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (4, 4), (5, 150, 20)).save(texture_path)

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))

    assert (
        add_ovrtx_preview_fallbacks_for_texture_file_materials(
            stage,
            diffuse_color_primvar="displayColor",
        )
        == 1
    )

    preview = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Painted/OVRTXPreviewSurface"),
    )
    sources, _ = preview.GetInput("diffuseColor").GetConnectedSources()
    assert sources[0].source.GetPrim().GetName() == "OVRTXPreviewDisplayColorReader"
    reader = UsdShade.Shader(sources[0].source.GetPrim())
    assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float3"
    assert reader.GetInput("varname").Get() == "displayColor"
    assert reader.GetInput("fallback").Get() == pytest.approx(
        (5 / 255, 150 / 255, 20 / 255),
    )


def test_textured_ovrtx_preview_fallback_rejects_conflicting_modes() -> None:
    stage = Usd.Stage.CreateInMemory()

    with pytest.raises(ValueError, match="mutually exclusive"):
        add_ovrtx_preview_fallbacks_for_texture_file_materials(
            stage,
            connect_diffuse_texture=True,
            diffuse_color_primvar="displayColor",
        )


def test_localizes_usdz_package_texture_assets_for_render(tmp_path: Path) -> None:
    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (4, 4), (5, 150, 20)).save(texture_path)
    package_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package_path, "w") as package:
        package.write(texture_path, "0/painted_albedo.png")

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    attr = material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    )
    attr.Set(Sdf.AssetPath(f"{package_path}[0/painted_albedo.png]"))

    assert (
        localize_package_texture_assets_for_render(stage, tmp_path / "localized") == 1
    )

    localized_asset = attr.Get()
    localized_path = Path(localized_asset.path)
    assert localized_path.is_file()
    assert localized_path.name == "painted_albedo.png"
    with Image.open(localized_path) as image:
        assert image.getpixel((0, 0)) == (5, 150, 20)


def test_localize_package_texture_assets_skips_oversized_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package_path, "w") as package:
        package.writestr("0/painted_albedo.png", b"abcd")

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    attr = material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    )
    attr.Set(Sdf.AssetPath(f"{package_path}[0/painted_albedo.png]"))
    monkeypatch.setattr(material_utils, "_MAX_PACKAGE_TEXTURE_BYTES", 3)

    assert (
        localize_package_texture_assets_for_render(stage, tmp_path / "localized") == 0
    )

    assert attr.Get().path == f"{package_path}[0/painted_albedo.png]"
    assert not (
        tmp_path
        / "localized"
        / material_utils._localized_package_texture_root(package_path)
        / "0"
        / "painted_albedo.png"
    ).exists()


def test_localizes_package_texture_assets_after_disabling_regular_instance(
    tmp_path: Path,
) -> None:
    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (4, 4), (90, 45, 10)).save(texture_path)
    package_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package_path, "w") as package:
        package.write(texture_path, "0/painted_albedo.png")

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().SetInstanceable(True)
    attr = material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    )
    attr.Set(Sdf.AssetPath(f"{package_path}[0/painted_albedo.png]"))

    assert (
        localize_package_texture_assets_for_render(stage, tmp_path / "localized") == 1
    )

    assert material.GetPrim().IsInstanceable() is False
    localized_path = Path(attr.Get().path)
    assert localized_path.is_file()
    with Image.open(localized_path) as image:
        assert image.getpixel((0, 0)) == (90, 45, 10)


def test_localizes_relative_usdz_package_texture_assets_for_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (4, 4), (80, 40, 200)).save(texture_path)
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    package_path = stage_dir / "asset.usdz"
    with zipfile.ZipFile(package_path, "w") as package:
        package.write(texture_path, "0/painted_albedo.png")

    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    stage_path = stage_dir / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    attr = material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    )
    attr.Set(Sdf.AssetPath("asset.usdz[0/painted_albedo.png]"))
    stage.GetRootLayer().Save()

    assert (
        localize_package_texture_assets_for_render(stage, tmp_path / "localized") == 1
    )

    localized_path = Path(attr.Get().path)
    assert localized_path.is_file()
    with Image.open(localized_path) as image:
        assert image.getpixel((0, 0)) == (80, 40, 200)


def test_localized_usdz_package_textures_do_not_collide_by_stem(
    tmp_path: Path,
) -> None:
    first_texture = tmp_path / "first.png"
    second_texture = tmp_path / "second.png"
    Image.new("RGB", (4, 4), (5, 150, 20)).save(first_texture)
    Image.new("RGB", (4, 4), (200, 40, 80)).save(second_texture)

    first_package = tmp_path / "a" / "asset.usdz"
    second_package = tmp_path / "b" / "asset.usdz"
    first_package.parent.mkdir()
    second_package.parent.mkdir()
    with zipfile.ZipFile(first_package, "w") as package:
        package.write(first_texture, "0/painted_albedo.png")
    with zipfile.ZipFile(second_package, "w") as package:
        package.write(second_texture, "0/painted_albedo.png")

    stage = Usd.Stage.CreateInMemory()
    first_material = UsdShade.Material.Define(stage, "/World/Looks/First")
    first_attr = first_material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    )
    first_attr.Set(Sdf.AssetPath(f"{first_package}[0/painted_albedo.png]"))
    second_material = UsdShade.Material.Define(stage, "/World/Looks/Second")
    second_attr = second_material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    )
    second_attr.Set(Sdf.AssetPath(f"{second_package}[0/painted_albedo.png]"))

    assert (
        localize_package_texture_assets_for_render(stage, tmp_path / "localized") == 2
    )

    first_localized = Path(first_attr.Get().path)
    second_localized = Path(second_attr.Get().path)
    assert first_localized != second_localized
    assert (
        first_localized.relative_to(tmp_path / "localized")
        .parts[0]
        .startswith("asset-")
    )
    assert (
        second_localized.relative_to(tmp_path / "localized")
        .parts[0]
        .startswith("asset-")
    )
    with Image.open(first_localized) as image:
        assert image.getpixel((0, 0)) == (5, 150, 20)
    with Image.open(second_localized) as image:
        assert image.getpixel((0, 0)) == (200, 40, 80)


def test_bakes_texture_file_material_to_mesh_display_color_for_render(
    tmp_path: Path,
) -> None:
    from pxr import Gf, Vt

    texture_path = tmp_path / "painted_albedo.png"
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 255, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((1, 1), (255, 255, 0))
    image.save(texture_path)

    stage_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "faceVarying",
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0, 0),
                Gf.Vec2f(1, 0),
                Gf.Vec2f(1, 1),
                Gf.Vec2f(0, 1),
            ],
        ),
    )
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    assert bake_texture_file_materials_to_display_color_for_render(stage) == 1

    display_color = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("displayColor")
    assert display_color.GetInterpolation() == "faceVarying"
    assert list(display_color.Get()) == pytest.approx(
        [
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
    )


def test_display_color_bake_skips_non_finite_uv_samples(tmp_path: Path) -> None:
    from pxr import Gf, Vt

    texture_path = tmp_path / "painted_albedo.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(texture_path)

    stage_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "faceVarying",
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0, 0),
                Gf.Vec2f(float("nan"), 0),
                Gf.Vec2f(0, 1),
            ],
        ),
    )
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    assert bake_texture_file_materials_to_display_color_for_render(stage) == 0
    display_color = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("displayColor")
    assert not display_color.HasValue()


def test_ovrtx_preview_fallback_overlay_covers_sublayered_materials(
    tmp_path: Path,
) -> None:
    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    _create_materialx_openpbr_material(material_stage)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"

    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    material = UsdShade.Material(combined.GetPrimAtPath("/World/Looks/Gold"))
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()

    original = Usd.Stage.Open(str(material_layer_path))
    assert original is not None
    original_material = UsdShade.Material(
        original.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert _connected_surface_shader_id(original_material) is None
    assert original_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_overlay_disables_referenced_material_instance(
    tmp_path: Path,
) -> None:
    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    _create_materialx_openpbr_material(prototype_stage, "/Prototype/Gold")
    prototype_stage.GetRootLayer().Save()

    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    material = UsdShade.Material.Define(material_stage, "/World/Looks/Gold")
    material.GetPrim().GetReferences().AddReference(
        str(prototype_path),
        "/Prototype/Gold",
    )
    material.GetPrim().SetInstanceable(True)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Gold"))
    assert material.GetPrim().IsInstance()

    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"
    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    combined_material = UsdShade.Material(
        combined.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert not combined_material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(combined_material) == "UsdPreviewSurface"
    assert not combined_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_overlay_suppresses_existing_surface_materialx(
    tmp_path: Path,
) -> None:
    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    material = _create_materialx_openpbr_material(material_stage)
    _connect_existing_preview_surface(material)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"

    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    combined_material = UsdShade.Material(
        combined.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert _connected_surface_shader_id(combined_material) == "UsdPreviewSurface"
    assert not combined_material.GetSurfaceOutput("mtlx").HasConnectedSource()

    original = Usd.Stage.Open(str(material_layer_path))
    assert original is not None
    original_material = UsdShade.Material(
        original.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert _connected_surface_shader_id(original_material) == "UsdPreviewSurface"
    assert original_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ovrtx_preview_fallback_overlay_suppresses_referenced_surface_instance(
    tmp_path: Path,
) -> None:
    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    prototype_material = _create_materialx_openpbr_material(
        prototype_stage,
        "/Prototype/Gold",
    )
    _connect_existing_preview_surface(prototype_material)
    prototype_stage.GetRootLayer().Save()

    material_layer_path = tmp_path / "materials.usda"
    material_stage = Usd.Stage.CreateNew(str(material_layer_path))
    material = UsdShade.Material.Define(material_stage, "/World/Looks/Gold")
    material.GetPrim().GetReferences().AddReference(
        str(prototype_path),
        "/Prototype/Gold",
    )
    material.GetPrim().SetInstanceable(True)
    material_stage.GetRootLayer().Save()

    root_layer_path = tmp_path / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_layer_path))
    root_layer.subLayerPaths = [str(material_layer_path)]
    root_layer.Save()

    stage = Usd.Stage.Open(str(root_layer_path))
    assert stage is not None
    material = UsdShade.Material(stage.GetPrimAtPath("/World/Looks/Gold"))
    assert material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert material.GetSurfaceOutput("mtlx").HasConnectedSource()

    overlay_path = tmp_path / "ovrtx_material_fallbacks.usda"
    assert (
        write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            overlay_path,
        )
        == 1
    )

    combined_path = tmp_path / "combined.usda"
    combined_layer = Sdf.Layer.CreateNew(str(combined_path))
    combined_layer.subLayerPaths = [str(overlay_path), str(root_layer_path)]
    combined_layer.Save()

    combined = Usd.Stage.Open(str(combined_path))
    assert combined is not None
    combined_material = UsdShade.Material(
        combined.GetPrimAtPath("/World/Looks/Gold"),
    )
    assert not combined_material.GetPrim().IsInstance()
    assert _connected_surface_shader_id(combined_material) == "UsdPreviewSurface"
    assert not combined_material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_default_material_library_gets_ovrtx_preview_fallbacks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "apps"
        / "material_agent"
        / "data"
        / "materials"
        / "material_libs_default"
        / "materials_libs_v2.usd"
    )
    exported = tmp_path / "materials_libs_v2.usda"
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    assert stage.GetRootLayer().Export(str(exported))

    before = Usd.Stage.Open(str(exported))
    assert before is not None
    material_count = sum(1 for prim in before.Traverse() if prim.IsA(UsdShade.Material))

    assert add_ovrtx_preview_fallbacks_to_stage_file(exported) == material_count

    after = Usd.Stage.Open(str(exported))
    assert after is not None
    for prim in after.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        material = UsdShade.Material(prim)
        assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
        assert not material.GetSurfaceOutput("mtlx").HasConnectedSource()


def test_ensure_looks_scope_spec_types_existing_untyped_looks_only() -> None:
    layer = Sdf.Layer.CreateAnonymous()
    looks_spec = Sdf.CreatePrimInLayer(layer, "/Root/Looks")
    looks_spec.specifier = Sdf.SpecifierDef
    materials_spec = Sdf.CreatePrimInLayer(layer, "/Root/Materials")
    materials_spec.specifier = Sdf.SpecifierDef

    ensure_looks_scope_spec(layer, "/Root/Looks")
    ensure_looks_scope_spec(layer, "/Root/Materials")

    assert looks_spec.typeName == "Scope"
    assert not materials_spec.typeName


def test_ensure_looks_scope_spec_requires_opt_in_for_over_specs() -> None:
    layer = Sdf.Layer.CreateAnonymous()
    looks_spec = Sdf.CreatePrimInLayer(layer, "/Root/Looks")

    ensure_looks_scope_spec(layer, "/Root/Looks")
    assert not looks_spec.typeName

    ensure_looks_scope_spec(layer, "/Root/Looks", allow_over=True)
    assert looks_spec.typeName == "Scope"


def test_ensure_looks_scope_preserves_existing_looks_specifier() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")
    layer = stage.GetEditTarget().GetLayer()
    looks_spec = Sdf.CreatePrimInLayer(layer, "/Root/Looks")
    assert looks_spec.specifier == Sdf.SpecifierOver

    ensure_looks_scope(stage, "/Root/Looks/PhysMat")

    assert looks_spec.typeName == "Scope"
    assert looks_spec.specifier == Sdf.SpecifierOver


def test_ensure_looks_scope_defines_missing_looks_scope() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")

    ensure_looks_scope(stage, "/Root/Looks/PhysMat")

    looks_spec = stage.GetEditTarget().GetLayer().GetPrimAtPath("/Root/Looks")
    assert looks_spec.typeName == "Scope"
    assert looks_spec.specifier == Sdf.SpecifierDef


def test_ensure_looks_scope_types_all_looks_ancestors() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")
    layer = stage.GetEditTarget().GetLayer()
    outer_looks = Sdf.CreatePrimInLayer(layer, "/Root/Looks")
    inner_looks = Sdf.CreatePrimInLayer(layer, "/Root/Looks/Asset/Looks")

    ensure_looks_scope(stage, "/Root/Looks/Asset/Looks/PhysMat")

    assert inner_looks.typeName == "Scope"
    assert outer_looks.typeName == "Scope"
