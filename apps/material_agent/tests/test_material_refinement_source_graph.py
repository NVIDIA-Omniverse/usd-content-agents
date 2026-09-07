# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for source-backed material refinement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from world_understanding.functions.graphics.mock_rendering import MockRenderingBackend

import material_agent.material_refinement.source_graph as source_graph_module
from material_agent.material_library_generation import (
    source_graph as material_library_source_graph,
)
from material_agent.material_library_generation.authoring import (
    MaterialAuthoringOperation,
    MaterialAuthoringRequest,
    MaterialPackageAuthoringError,
    MaterialRecipeSemantics,
    SourceMaterialReference,
    author_material_package,
    load_material_package,
)
from material_agent.material_library_generation.schema import (
    MaterialRecipe,
    PBRHints,
    TextureMapSet,
)
from material_agent.material_refinement import (
    MaterialRefinementConfig,
    MaterialVariationConfig,
    TextureVariationArtifacts,
    TextureVariationRequest,
    run_material_refinement,
    run_material_variations,
)
from material_agent.material_refinement.source_graph import (
    MaterialGraphEdit,
    MaterialGraphEditError,
    MaterialGraphInfo,
    inspect_material_graph,
    write_edited_material_graphs,
)

_APPROVE = """**Critique:**
The material matches the goal.
**Score:** 9
**Decision:** APPROVE
**Improvement Suggestions:**
None.
"""


def test_omnipbr_asset_input_discovery_rejects_a_connection_cycle(
    tmp_path: Path,
) -> None:
    class _FakeConnectable:
        shader_input: Any

        def GetInput(self, _name: str) -> Any:  # noqa: N802 - USD-shaped stub
            return self.shader_input

    class _FakeInput:
        def __init__(self) -> None:
            self.connectable = _FakeConnectable()
            self.connectable.shader_input = self

        def GetAttr(self) -> Any:  # noqa: N802 - USD-shaped stub
            return SimpleNamespace(GetPath=lambda: "/Cycle.inputs:texture")

        def Get(self) -> None:  # noqa: N802 - USD-shaped stub
            return None

        def GetConnectedSources(self) -> tuple[list[Any], list[Any]]:  # noqa: N802
            return (
                [
                    SimpleNamespace(
                        sourceType=UsdShade.AttributeType.Input,
                        source=self.connectable,
                        sourceName="texture",
                    )
                ],
                [],
            )

    assert (
        material_library_source_graph._asset_path_from_input(  # noqa: SLF001
            _FakeInput(),  # type: ignore[arg-type]
            source_usd=tmp_path / "cycle.usda",
        )
        is None
    )


class _ApproveVlm:
    def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
        if str(kwargs.get("system_prompt", "")).startswith(
            "You infer normalized PBR controls"
        ):
            prompt = str(kwargs["final_prompt"])
            encoded_bounds = prompt.split("Allowed target ranges:\n", 1)[1].split(
                "\n\nReturn exactly", 1
            )[0]
            bounds = json.loads(encoded_bounds)

            def midpoint(name: str) -> float:
                return (bounds[name]["min"] + bounds[name]["max"]) / 2.0

            return json.dumps(
                {
                    "base_color": [
                        midpoint("base_color_r"),
                        midpoint("base_color_g"),
                        midpoint("base_color_b"),
                    ],
                    "roughness": midpoint("roughness"),
                    "metallic": midpoint("metallic"),
                    "reasoning": "Deterministic prompt-target test inference.",
                }
            )
        return _APPROVE


class _TextureGenerator:
    name = "source-graph-texture-fake"

    def generate(
        self,
        request: TextureVariationRequest,
        *,
        output_dir: Path,
        cancel_check: Any = None,
    ) -> TextureVariationArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        albedo = output_dir / "albedo.png"
        normal = output_dir / "normal.png"
        orm = output_dir / "orm.png"
        Image.new("RGB", (8, 8), (102, 76, 153)).save(albedo)
        Image.new("RGB", (8, 8), (128, 128, 255)).save(normal)
        Image.new("RGB", (8, 8), (255, 140, 230)).save(orm)
        return TextureVariationArtifacts(
            albedo_path=albedo,
            normal_path=normal,
            orm_path=orm,
            variant_asset_uri=request.source_asset_path.as_uri(),
            metadata={"seed": request.seed, "strength": request.strength},
            diagnostics=(),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_scalar_openpbr(path: Path, *, preview_fallback: bool = False) -> str:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material_path = "/World/Looks/ScalarOpenPBR"
    material = UsdShade.Material.Define(stage, material_path)
    base_color = material.CreateInput("base_color", Sdf.ValueTypeNames.Color3f)
    base_color.Set(Gf.Vec3f(0.2, 0.4, 0.7))
    roughness = material.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float)
    roughness.Set(0.25)
    metallic = material.CreateInput("base_metalness", Sdf.ValueTypeNames.Float)
    metallic.Set(1.0)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/OpenPBR")
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        base_color
    )
    shader.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness
    )
    shader.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).ConnectToSource(
        metallic
    )
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    if preview_fallback:
        preview = UsdShade.Shader.Define(stage, f"{material_path}/OVRTXPreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.2, 0.4, 0.7)
        )
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(
            preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
    stage.GetRootLayer().Save()
    return material_path


def _create_scalar_preview_surface(path: Path) -> str:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material_path = "/World/Looks/ScalarPreview"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.15, 0.3, 0.55)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()
    return material_path


def _create_scalar_omnipbr_mdl(
    path: Path,
    *,
    source_asset: str = "OmniPBR.mdl",
    sub_identifier: str = "OmniPBR",
    texture_path: str | None = None,
) -> str:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material_path = "/World/Looks/ScalarOmniPBR"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.GetPrim().CreateAttribute(
        "info:implementationSource", Sdf.ValueTypeNames.Token
    ).Set("sourceAsset")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(source_asset))
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
    ).Set(sub_identifier)
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.28, 0.30, 0.32)
    )
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
        0.38
    )
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture_path is not None:
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_path)
        )
    shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader_output)
    stage.GetRootLayer().Save()
    return material_path


def _create_textures(directory: Path, *, color: tuple[int, int, int]) -> TextureMapSet:
    directory.mkdir(parents=True, exist_ok=True)
    textures = TextureMapSet(
        albedo=directory / "albedo.png",
        normal=directory / "normal.png",
        orm=directory / "orm.png",
    )
    Image.new("RGB", (8, 8), color).save(textures.albedo)
    Image.new("RGB", (8, 8), (128, 128, 255)).save(textures.normal)
    Image.new("RGB", (8, 8), (255, 128, 230)).save(textures.orm)
    return textures


def _image_shader(
    stage: Usd.Stage,
    path: str,
    *,
    shader_id: str,
    texture: Path,
    output_type: Any,
) -> UsdShade.Shader:
    shader = UsdShade.Shader.Define(stage, path)
    shader.CreateIdAttr(shader_id)
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture.name)
    )
    shader.CreateOutput("out", output_type)
    return shader


def _create_textured_openpbr(path: Path) -> tuple[str, TextureMapSet]:
    textures = _create_textures(path.parent, color=(64, 96, 160))
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material_path = "/World/Looks/TexturedOpenPBR"
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/OpenPBR")
    surface.CreateIdAttr("ND_open_pbr_surface_surfaceshader")

    albedo = _image_shader(
        stage,
        f"{material_path}/Albedo",
        shader_id="ND_tiledimage_color3",
        texture=textures.albedo,
        output_type=Sdf.ValueTypeNames.Color3f,
    )
    surface.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        albedo.GetOutput("out")
    )

    normal = _image_shader(
        stage,
        f"{material_path}/Normal",
        shader_id="ND_tiledimage_vector3",
        texture=textures.normal,
        output_type=Sdf.ValueTypeNames.Float3,
    )
    normal_map = UsdShade.Shader.Define(stage, f"{material_path}/NormalMap")
    normal_map.CreateIdAttr("ND_normalmap")
    normal_map.CreateInput("in", Sdf.ValueTypeNames.Float3).ConnectToSource(
        normal.GetOutput("out")
    )
    normal_map.CreateOutput("out", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("geometry_normal", Sdf.ValueTypeNames.Float3).ConnectToSource(
        normal_map.GetOutput("out")
    )

    orm = _image_shader(
        stage,
        f"{material_path}/ORM",
        shader_id="ND_tiledimage_color3",
        texture=textures.orm,
        output_type=Sdf.ValueTypeNames.Color3f,
    )
    separate = UsdShade.Shader.Define(stage, f"{material_path}/SeparateORM")
    separate.CreateIdAttr("ND_separate3_color3")
    separate.CreateInput("in", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        orm.GetOutput("out")
    )
    separate.CreateOutput("outg", Sdf.ValueTypeNames.Float)
    separate.CreateOutput("outb", Sdf.ValueTypeNames.Float)
    surface.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        separate.GetOutput("outg")
    )
    surface.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).ConnectToSource(
        separate.GetOutput("outb")
    )
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        surface.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()
    return material_path, textures


def test_source_graph_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    class BrokenMdlShader:
        def GetSourceAsset(self, _context: str):
            raise RuntimeError("broken source asset")

    class BrokenPort:
        def GetConnectedSources(self):
            raise RuntimeError("broken connection")

    assert source_graph_module._mdl_source_identity(None) == ""
    assert source_graph_module._mdl_source_identity(BrokenMdlShader()) == ""
    assert source_graph_module._single_connected_source(BrokenPort()) is None
    assert (
        source_graph_module._single_connected_source(
            SimpleNamespace(GetConnectedSources=lambda: ([], []))
        )
        is None
    )
    assert source_graph_module._input_sources(BrokenPort()) == []
    assert source_graph_module._boolean_value(1, label="thin_walled") is True

    class FakeInput:
        def __init__(self, path: str) -> None:
            self.path = path
            self.sources: list[Any] = []

        def GetAttr(self):
            return SimpleNamespace(GetPath=lambda: Sdf.Path(self.path))

        def GetConnectedSources(self):
            return self.sources, []

    loop_input = FakeInput("/Loop.inputs:in")
    loop_input.sources = [
        SimpleNamespace(
            sourceType=UsdShade.AttributeType.Input,
            source=SimpleNamespace(GetInput=lambda _name: loop_input),
            sourceName="in",
        )
    ]
    assert source_graph_module._find_upstream_image(loop_input) is None

    unsupported_input = FakeInput("/Unsupported.inputs:in")
    unsupported_input.sources = [SimpleNamespace(sourceType=object())]
    assert source_graph_module._find_upstream_image(unsupported_input) is None

    invalid_shader_input = FakeInput("/InvalidShader.inputs:in")
    invalid_shader_input.sources = [
        SimpleNamespace(
            sourceType=UsdShade.AttributeType.Output,
            source=SimpleNamespace(GetPrim=Usd.Prim),
        )
    ]
    assert source_graph_module._find_upstream_image(invalid_shader_input) is None

    source_usd = tmp_path / "source.usda"
    source_usd.touch()
    uri_shader = SimpleNamespace(
        GetInput=lambda _name: SimpleNamespace(
            Get=lambda: Sdf.AssetPath("https://example.test/albedo.png")
        )
    )
    assert (
        source_graph_module._asset_path_from_image(uri_shader, source_usd=source_usd)
        is None
    )
    relative_shader = SimpleNamespace(
        GetInput=lambda _name: SimpleNamespace(Get=lambda: Sdf.AssetPath("missing.png"))
    )
    assert (
        source_graph_module._asset_path_from_image(
            relative_shader, source_usd=source_usd
        )
        is None
    )
    assert (
        source_graph_module._material_input_value(
            SimpleNamespace(GetInput=lambda _name: None), ("missing",)
        )
        is None
    )
    assert (
        source_graph_module._shader_input_value(
            SimpleNamespace(GetInput=lambda _name: None), ("missing",)
        )
        is None
    )
    assert (
        source_graph_module._resolved_asset_file(
            Sdf.AssetPath("https://example.test/asset.png"), source_usd
        )
        is None
    )

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdShade.Material.Define(stage, "/World/Looks/Test")
    source_graph_module._restore_captured_assets(
        stage,
        (
            source_graph_module._CapturedAsset(
                relative_prim_path="",
                attribute_name="missing",
                values=(Sdf.AssetPath("missing.png"),),
                is_array=False,
            ),
        ),
        source_usd=source_usd,
        target_material_path=Sdf.Path("/World/Looks/Test"),
        output_usd=tmp_path / "output.usda",
    )
    with pytest.raises(MaterialGraphEditError, match="missing expected prim"):
        source_graph_module._restore_captured_assets(
            stage,
            (
                source_graph_module._CapturedAsset(
                    relative_prim_path="Missing",
                    attribute_name="asset",
                    values=(Sdf.AssetPath("missing.png"),),
                    is_array=False,
                ),
            ),
            source_usd=source_usd,
            target_material_path=Sdf.Path("/World/Looks/Test"),
            output_usd=tmp_path / "output.usda",
        )


@pytest.mark.parametrize(
    ("channel", "suffix", "image_format"),
    (("albedo", ".png", "PNG"), ("orm", ".tiff", "TIFF")),
)
def test_source_feature_extraction_rejects_non_8bit_maps(
    tmp_path: Path,
    channel: str,
    suffix: str,
    image_format: str,
) -> None:
    textures = _create_textures(tmp_path, color=(64, 96, 160))
    high_depth = tmp_path / f"{channel}_16bit{suffix}"
    Image.new("I;16", (8, 8), 32768).save(high_depth, format=image_format)
    textures = TextureMapSet(
        albedo=high_depth if channel == "albedo" else textures.albedo,
        normal=textures.normal,
        orm=high_depth if channel == "orm" else textures.orm,
    )

    with pytest.raises(
        MaterialGraphEditError,
        match=rf"source {channel} texture must use 8-bit channels",
    ):
        source_graph_module._mean_texture_features(textures)


def test_source_graph_rejects_connections_outside_material_scope(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "external-connection.usda"
    stage = Usd.Stage.CreateNew(str(source_usd))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material_path = "/World/Looks/ExternalConnection"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, "/World/SharedShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.4, 0.7)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()

    with pytest.raises(MaterialGraphEditError, match="outside the material scope"):
        write_edited_material_graphs(
            tmp_path / "candidate.usda",
            (
                MaterialGraphEdit(
                    source_usd=source_usd,
                    source_material_prim_path=material_path,
                    target_material_prim_path="/World/Looks/Edited",
                ),
            ),
        )


def test_openpbr_scalar_values_may_live_on_surface_shader(tmp_path: Path) -> None:
    source_usd = tmp_path / "shader-values.usda"
    stage = Usd.Stage.CreateNew(str(source_usd))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material_path = "/World/Looks/ShaderValues"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/OpenPBR")
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    shader.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.1, 0.3, 0.7)
    )
    shader.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    shader.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()

    info = inspect_material_graph(source_usd, material_path)

    assert info.base_color == pytest.approx((0.1, 0.3, 0.7))
    assert info.roughness == pytest.approx(0.45)
    assert info.metallic == pytest.approx(0.0)


def test_asset_arrays_are_preserved_when_existing_output_is_replaced(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_preview_surface(source_usd)
    source_stage = Usd.Stage.Open(str(source_usd))
    source_stage.GetPrimAtPath(material_path).CreateAttribute(
        "custom:assetArray", Sdf.ValueTypeNames.AssetArray
    ).Set(Sdf.AssetPathArray([Sdf.AssetPath("https://example.test/reference.png")]))
    source_stage.GetRootLayer().Save()
    output_usd = tmp_path / "candidate.usda"
    edit = MaterialGraphEdit(
        source_usd=source_usd,
        source_material_prim_path=material_path,
        target_material_prim_path="/World/Looks/Edited",
    )

    write_edited_material_graphs(output_usd, (edit,))
    write_edited_material_graphs(output_usd, (edit,))

    output_stage = Usd.Stage.Open(str(output_usd))
    assets = output_stage.GetPrimAtPath("/World/Looks/Edited").GetAttribute(
        "custom:assetArray"
    )
    assert list(assets.Get()) == [Sdf.AssetPath("https://example.test/reference.png")]


def test_source_graph_rejects_instance_proxy_material(tmp_path: Path) -> None:
    prototype_usd = tmp_path / "prototype.usda"
    _create_scalar_openpbr(prototype_usd)
    source_usd = tmp_path / "instanced.usda"
    stage = Usd.Stage.CreateNew(str(source_usd))
    instance = UsdGeom.Xform.Define(stage, "/World/Asset").GetPrim()
    instance.GetReferences().AddReference(str(prototype_usd), "/World")
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()
    proxy_path = "/World/Asset/Looks/ScalarOpenPBR"

    assert stage.GetPrimAtPath(proxy_path).IsInstanceProxy()
    with pytest.raises(MaterialGraphEditError, match="read-only instance proxy"):
        inspect_material_graph(source_usd, proxy_path)


def test_source_graph_disables_instanceable_material_before_editing(
    tmp_path: Path,
) -> None:
    prototype_usd = tmp_path / "prototype.usda"
    material_path = _create_scalar_openpbr(prototype_usd)
    source_usd = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_usd))
    UsdGeom.Scope.Define(source_stage, "/World")
    UsdGeom.Scope.Define(source_stage, "/World/Looks")
    source_material = UsdShade.Material.Define(source_stage, material_path)
    source_material.GetPrim().GetReferences().AddReference(
        str(prototype_usd), material_path
    )
    source_material.GetPrim().SetInstanceable(True)
    source_stage.GetRootLayer().Save()
    source_hash = _sha256(source_usd)
    output_usd = tmp_path / "candidate.usda"
    target_path = "/World/Looks/Edited"

    write_edited_material_graphs(
        output_usd,
        (
            MaterialGraphEdit(
                source_usd=source_usd,
                source_material_prim_path=material_path,
                target_material_prim_path=target_path,
                base_color=(0.6, 0.2, 0.1),
                roughness=0.5,
                metallic=0.0,
            ),
        ),
    )

    output_stage = Usd.Stage.Open(str(output_usd))
    assert _sha256(source_usd) == source_hash
    assert not output_stage.GetPrimAtPath(target_path).IsInstanceable()


def test_source_contract_derives_values_from_openpbr_graph(tmp_path: Path) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_openpbr(source_usd)

    config = MaterialRefinementConfig.from_mapping(
        {
            "goal": {"appearance_prompt": "satin blue metal"},
            "source": {
                "name": "Source OpenPBR",
                "source_usd": source_usd.name,
                "material_prim_path": material_path,
                "material_profile": "openpbr_materialx",
            },
            "output_dir": "output",
        },
        base_dir=tmp_path,
    )

    assert config.source.is_graph_backed is True
    assert config.source.material_profile == "openpbr_materialx"
    assert config.source.representation == "scalar_pbr"
    assert config.source.base_color == pytest.approx((0.2, 0.4, 0.7))
    assert config.source.roughness == pytest.approx(0.25)
    assert config.source.metallic == pytest.approx(1.0)


def test_source_contract_derives_optical_values_from_openpbr_graph(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "optical-source.usda"
    material_path = _create_scalar_openpbr(source_usd)
    stage = Usd.Stage.Open(str(source_usd))
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/OpenPBR"))
    optical_inputs = (
        ("geometry_opacity", Sdf.ValueTypeNames.Float, 0.63),
        ("transmission_weight", Sdf.ValueTypeNames.Float, 0.72),
        ("specular_ior", Sdf.ValueTypeNames.Float, 1.33),
        ("geometry_thin_walled", Sdf.ValueTypeNames.Bool, True),
    )
    for name, value_type, value in optical_inputs:
        material.CreateInput(name, value_type).Set(value)
        shader.CreateInput(name, value_type).Set(value)
    stage.GetRootLayer().Save()

    config = MaterialRefinementConfig.from_mapping(
        {
            "goal": {"appearance_prompt": "translucent satin blue polymer"},
            "source": {
                "source_usd": source_usd.name,
                "material_prim_path": material_path,
            },
            "output_dir": "output",
        },
        base_dir=tmp_path,
    )

    assert config.source.opacity == pytest.approx(0.63)
    assert config.source.transmission == pytest.approx(0.72)
    assert config.source.ior == pytest.approx(1.33)
    assert config.source.thin_walled is True


def test_source_contract_derives_values_from_omnipbr_mdl_graph(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_omnipbr_mdl(source_usd)

    config = MaterialRefinementConfig.from_mapping(
        {
            "goal": {"appearance_prompt": "deep green satin polymer"},
            "source": {
                "name": "Source OmniPBR",
                "source_usd": source_usd.name,
                "material_prim_path": material_path,
                "material_profile": "omnipbr_mdl",
            },
            "output_dir": "output",
        },
        base_dir=tmp_path,
    )

    assert config.source.is_graph_backed is True
    assert config.source.material_profile == "omnipbr_mdl"
    assert config.source.representation == "scalar_pbr"
    assert config.source.base_color == pytest.approx((0.28, 0.30, 0.32))
    assert config.source.roughness == pytest.approx(0.38)
    assert config.source.metallic == pytest.approx(0.0)


def test_scalar_openpbr_edit_preserves_source_and_topology(tmp_path: Path) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_openpbr(source_usd, preview_fallback=True)
    source_info = inspect_material_graph(source_usd, material_path)
    source_hash = _sha256(source_usd)
    output_usd = tmp_path / "candidate" / "material_library.usda"

    (candidate,) = write_edited_material_graphs(
        output_usd,
        (
            MaterialGraphEdit(
                source_usd=source_usd,
                source_material_prim_path=material_path,
                target_material_prim_path="/World/Looks/EditedOpenPBR",
                base_color=(0.7, 0.3, 0.1),
                roughness=0.65,
                metallic=1.0,
            ),
        ),
    )

    assert _sha256(source_usd) == source_hash
    assert candidate.material_profile == "openpbr_materialx"
    assert candidate.representation == "scalar_pbr"
    assert candidate.base_color == pytest.approx((0.7, 0.3, 0.1))
    assert candidate.roughness == pytest.approx(0.65)
    assert candidate.metallic == pytest.approx(1.0)
    assert candidate.topology_sha256 == source_info.topology_sha256
    output_stage = Usd.Stage.Open(str(output_usd))
    fallback = UsdShade.Shader(
        output_stage.GetPrimAtPath("/World/Looks/EditedOpenPBR/OVRTXPreviewSurface")
    )
    assert tuple(fallback.GetInput("diffuseColor").Get()) == pytest.approx(
        (0.7, 0.3, 0.1)
    )
    assert fallback.GetInput("roughness").Get() == pytest.approx(0.65)
    assert fallback.GetInput("metallic").Get() == pytest.approx(1.0)


def test_scalar_preview_edit_preserves_profile_and_topology(tmp_path: Path) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_preview_surface(source_usd)
    source_info = inspect_material_graph(source_usd, material_path)

    (candidate,) = write_edited_material_graphs(
        tmp_path / "candidate.usda",
        (
            MaterialGraphEdit(
                source_usd=source_usd,
                source_material_prim_path=material_path,
                target_material_prim_path="/World/Looks/EditedPreview",
                base_color=(0.6, 0.2, 0.1),
                roughness=0.75,
                metallic=0.0,
            ),
        ),
    )

    assert candidate.material_profile == "preview_surface"
    assert candidate.representation == "scalar_pbr"
    assert candidate.base_color == pytest.approx((0.6, 0.2, 0.1))
    assert candidate.roughness == pytest.approx(0.75)
    assert candidate.topology_sha256 == source_info.topology_sha256


def test_scalar_omnipbr_mdl_edit_preserves_source_and_topology(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_omnipbr_mdl(source_usd)
    source_info = inspect_material_graph(source_usd, material_path)
    source_hash = _sha256(source_usd)
    output_usd = tmp_path / "candidate" / "material_library.usda"

    (candidate,) = write_edited_material_graphs(
        output_usd,
        (
            MaterialGraphEdit(
                source_usd=source_usd,
                source_material_prim_path=material_path,
                target_material_prim_path="/World/Looks/EditedOmniPBR",
                base_color=(0.04, 0.36, 0.12),
                roughness=0.52,
                metallic=0.0,
            ),
        ),
    )

    assert _sha256(source_usd) == source_hash
    assert source_info.shader_ids == ("mdl:OmniPBR.mdl#OmniPBR",)
    assert candidate.material_profile == "omnipbr_mdl"
    assert candidate.representation == "scalar_pbr"
    assert candidate.base_color == pytest.approx((0.04, 0.36, 0.12))
    assert candidate.roughness == pytest.approx(0.52)
    assert candidate.metallic == pytest.approx(0.0)
    assert candidate.shader_ids == source_info.shader_ids
    assert candidate.topology_sha256 == source_info.topology_sha256


def test_source_graph_rejects_non_omnipbr_mdl(tmp_path: Path) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_omnipbr_mdl(
        source_usd,
        source_asset="CustomMaterial.mdl",
        sub_identifier="CustomMaterial",
    )

    with pytest.raises(MaterialGraphEditError, match="unsupported MDL"):
        inspect_material_graph(source_usd, material_path)


def test_source_graph_rejects_textured_omnipbr_mdl(tmp_path: Path) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_omnipbr_mdl(
        source_usd,
        texture_path="albedo.png",
    )

    with pytest.raises(MaterialGraphEditError, match="texture-backed OmniPBR"):
        inspect_material_graph(source_usd, material_path)


def test_source_graph_treats_empty_omnipbr_asset_inputs_as_scalar(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_omnipbr_mdl(source_usd)
    stage = Usd.Stage.Open(str(source_usd))
    shader = UsdShade.Shader.Get(stage, f"{material_path}/Shader")
    for name in ("diffuse_texture", "normalmap_texture", "ORM_texture"):
        shader.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(""))
    stage.GetRootLayer().Save()

    info = inspect_material_graph(source_usd, material_path)

    assert info.representation == "scalar_pbr"
    assert info.textures is None


def test_textured_openpbr_edit_retargets_maps_without_rebuilding_graph(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_usd = source_dir / "source.usda"
    material_path, _source_textures = _create_textured_openpbr(source_usd)
    source_info = inspect_material_graph(source_usd, material_path)
    source_hash = _sha256(source_usd)
    replacements = _create_textures(tmp_path / "replacements", color=(179, 77, 26))
    output_usd = tmp_path / "candidate" / "material_library.usda"

    (candidate,) = write_edited_material_graphs(
        output_usd,
        (
            MaterialGraphEdit(
                source_usd=source_usd,
                source_material_prim_path=material_path,
                target_material_prim_path="/World/Looks/EditedTexturedOpenPBR",
                base_color=(179 / 255, 77 / 255, 26 / 255),
                roughness=128 / 255,
                metallic=230 / 255,
                textures=replacements,
            ),
        ),
    )

    assert _sha256(source_usd) == source_hash
    assert candidate.material_profile == "openpbr_materialx"
    assert candidate.representation == "textured_pbr"
    assert candidate.topology_sha256 == source_info.topology_sha256
    assert candidate.textures is not None
    assert all(
        path.is_relative_to(output_usd.parent)
        for path in (
            candidate.textures.albedo,
            candidate.textures.normal,
            candidate.textures.orm,
        )
    )
    assert candidate.base_color == pytest.approx(
        (0.45078578, 0.07421357, 0.01032982), abs=1.0e-6
    )


def test_textured_source_backed_refinement_preserves_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_usd = source_dir / "source.usda"
    material_path, _textures = _create_textured_openpbr(source_usd)
    source_info = inspect_material_graph(source_usd, material_path)
    source_hash = _sha256(source_usd)
    config = MaterialRefinementConfig.from_mapping(
        {
            "goal": {
                "appearance_prompt": "mottled violet metal",
            },
            "source": {
                "name": "Textured OpenPBR",
                "source_usd": source_usd.as_posix(),
                "material_prim_path": material_path,
            },
            "optimization": {
                "name": "random",
                "max_trials": 1,
                "seed": 42,
                "max_refinements": 0,
            },
            "render": {
                "backend": "mock",
                "image_width": 32,
                "image_height": 32,
                "camera_corners": ["+x+y+z"],
            },
            "judge": {
                "vlm": {"backend": "fake", "model": "fake"},
                "score_threshold": 0.7,
            },
            "output_dir": (tmp_path / "output").as_posix(),
        },
        base_dir=tmp_path,
    )

    import material_agent.material_refinement.runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "write_material_library_usd",
        lambda *args, **kwargs: pytest.fail(
            "source-backed refinement called the material generation writer"
        ),
    )

    result = run_material_refinement(
        config,
        generator=_TextureGenerator(),
        rendering_backend=MockRenderingBackend(),
        vlm_judge=_ApproveVlm(),
    )

    assert result.approved is True
    assert _sha256(source_usd) == source_hash
    source_copy = inspect_material_graph(
        config.output_dir / "source/material/material_library.usda",
        "/World/Looks/Textured_OpenPBR",
    )
    candidate = inspect_material_graph(
        result.best_artifacts["material_usd"],
        "/World/Looks/Textured_OpenPBR",
    )
    assert source_copy.representation == "textured_pbr"
    assert candidate.representation == "textured_pbr"
    assert source_copy.topology_sha256 == source_info.topology_sha256
    assert candidate.topology_sha256 == source_info.topology_sha256


def test_source_backed_refinement_never_calls_generation_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_openpbr(source_usd)
    source_hash = _sha256(source_usd)
    data = {
        "goal": {"appearance_prompt": "satin blue metal"},
        "source": {
            "name": "Source OpenPBR",
            "source_usd": source_usd.as_posix(),
            "material_prim_path": material_path,
            "material_profile": "openpbr_materialx",
        },
        "optimization": {
            "name": "random",
            "max_trials": 1,
            "seed": 42,
            "max_refinements": 0,
        },
        "render": {
            "backend": "mock",
            "image_width": 32,
            "image_height": 32,
            "camera_corners": ["+x+y+z"],
        },
        "judge": {
            "vlm": {"backend": "fake", "model": "fake"},
            "score_threshold": 0.7,
        },
        "output_dir": (tmp_path / "output").as_posix(),
    }
    config = MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)

    import material_agent.material_refinement.runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "write_material_library_usd",
        lambda *args, **kwargs: pytest.fail(
            "source-backed refinement called the material generation writer"
        ),
    )

    result = run_material_refinement(
        config,
        rendering_backend=MockRenderingBackend(),
        vlm_judge=_ApproveVlm(),
    )

    assert result.approved is True
    assert _sha256(source_usd) == source_hash
    candidate = inspect_material_graph(
        result.best_artifacts["material_usd"],
        "/World/Looks/Source_OpenPBR",
    )
    assert candidate.material_profile == "openpbr_materialx"
    assert (
        candidate.topology_sha256
        == inspect_material_graph(source_usd, material_path).topology_sha256
    )


def test_openpbr_refinement_can_cross_from_metal_to_dielectric_paint(
    tmp_path: Path,
) -> None:
    class PaintTargetVlm:
        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            if str(kwargs.get("system_prompt", "")).startswith(
                "You infer normalized PBR controls"
            ):
                return json.dumps(
                    {
                        "base_color": [0.02, 0.08, 0.42],
                        "roughness": 0.48,
                        "metallic": 0.0,
                        "reasoning": "Paint is dielectric rather than exposed metal.",
                    }
                )
            return _APPROVE

    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_openpbr(source_usd)
    source_info = inspect_material_graph(source_usd, material_path)
    config = MaterialRefinementConfig.from_mapping(
        {
            "goal": {
                "appearance_prompt": (
                    "Satin cobalt-blue painted steel with soft broad highlights "
                    "and a physically plausible dielectric response."
                )
            },
            "source": {
                "name": "Source OpenPBR",
                "source_usd": source_usd.as_posix(),
                "material_prim_path": material_path,
                "material_profile": "openpbr_materialx",
            },
            "optimization": {
                "name": "random",
                "max_trials": 1,
                "seed": 42,
                "max_refinements": 0,
            },
            "render": {
                "backend": "mock",
                "image_width": 32,
                "image_height": 32,
                "camera_corners": ["+x+y+z"],
            },
            "judge": {
                "vlm": {"backend": "fake", "model": "fake"},
                "score_threshold": 0.7,
            },
            "output_dir": (tmp_path / "output").as_posix(),
        },
        base_dir=tmp_path,
    )

    result = run_material_refinement(
        config,
        rendering_backend=MockRenderingBackend(),
        vlm_judge=PaintTargetVlm(),
    )

    assert result.approved is True
    assert result.best_params["metallic"] == pytest.approx(0.0)
    assert result.search_space.bounds("metallic") == pytest.approx((0.0, 1.0))
    target_evidence = json.loads(
        config.output_dir.joinpath("target_inference.json").read_text(encoding="utf-8")
    )
    assert target_evidence["source_search_space"]["metallic"] == {
        "integer": False,
        "max": 1.0,
        "min": 1.0,
    }
    assert "metallic" in target_evidence["initial_expanded_controls"]
    candidate = inspect_material_graph(
        result.best_artifacts["material_usd"],
        "/World/Looks/Source_OpenPBR",
    )
    assert candidate.metallic == pytest.approx(0.0)
    assert candidate.material_profile == "openpbr_materialx"
    assert candidate.topology_sha256 == source_info.topology_sha256


def test_source_backed_variations_publish_isolated_openpbr_graphs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    material_path = _create_scalar_openpbr(source_usd)
    source_hash = _sha256(source_usd)
    config = MaterialVariationConfig.from_mapping(
        {
            "goal": {
                "appearance_prompt": "satin blue metal",
            },
            "variation_set": {
                "description": "Two isolated OpenPBR treatments.",
                "requested_count": 2,
                "minimum_feature_distance": 0.0,
            },
            "source": {
                "name": "Source OpenPBR",
                "source_usd": source_usd.as_posix(),
                "material_prim_path": material_path,
                "material_profile": "openpbr_materialx",
            },
            "optimization": {
                "name": "random",
                "max_trials": 1,
                "seed": 42,
                "max_refinements": 0,
            },
            "render": {
                "backend": "mock",
                "image_width": 32,
                "image_height": 32,
                "camera_corners": ["+x+y+z"],
            },
            "judge": {
                "vlm": {"backend": "fake", "model": "fake"},
                "score_threshold": 0.7,
            },
            "output_dir": (tmp_path / "variations").as_posix(),
        },
        base_dir=tmp_path,
    )

    import material_agent.material_refinement.runner as runner_module
    import material_agent.material_refinement.variation as variation_module

    def fail_generation(*args: Any, **kwargs: Any) -> None:
        pytest.fail("source-backed variation called the material generation writer")

    monkeypatch.setattr(runner_module, "write_material_library_usd", fail_generation)
    monkeypatch.setattr(variation_module, "write_material_library_usd", fail_generation)

    result = run_material_variations(
        config,
        rendering_backend=MockRenderingBackend(),
        vlm_judge=_ApproveVlm(),
    )

    assert result.success is True
    assert result.material_library_path is not None
    assert _sha256(source_usd) == source_hash
    assert len(result.selected_materials) == 2
    bindings = [material.binding for material in result.selected_materials]
    assert len(set(bindings)) == 2
    infos = [
        inspect_material_graph(result.material_library_path, binding)
        for binding in bindings
    ]
    source_topology = inspect_material_graph(source_usd, material_path).topology_sha256
    assert all(info.material_profile == "openpbr_materialx" for info in infos)
    assert all(info.topology_sha256 == source_topology for info in infos)


def test_shared_authoring_core_creates_standalone_scalar_package(
    tmp_path: Path,
) -> None:
    recipe = MaterialRecipe(
        id="satin_blue",
        name="Satin Blue",
        description="A scalar satin blue coating.",
        appearance_prompt="satin blue painted coating",
        base_color_hint=(0.1, 0.3, 0.7),
        pbr_hints=PBRHints(roughness=0.42, metallic=0.0),
    )
    request = MaterialAuthoringRequest(
        operation=MaterialAuthoringOperation.CREATE,
        recipe=recipe,
        material_profile="preview_surface",
    )

    package = author_material_package(request, tmp_path / "standalone")
    loaded = load_material_package(tmp_path / "standalone")
    cached = author_material_package(request, tmp_path / "standalone")

    assert package.material_usd_path.is_file()
    assert package.materials_manifest_path.is_file()
    assert package.authoring_manifest_path.is_file()
    assert package.material_profile == "preview_surface"
    assert package.representation == "scalar_pbr"
    assert package.validation["cache_hit"] is False
    assert loaded.request_id == request.request_id
    assert loaded.validation["cache_hit"] is False
    assert package.validation["library_path"] == str(package.material_usd_path)
    assert cached.validation["cache_hit"] is True
    authoring_manifest = json.loads(
        package.authoring_manifest_path.read_text(encoding="utf-8")
    )
    assert (
        authoring_manifest["material_package"]["validation"]["library_path"]
        == "material_library.usda"
    )
    info = inspect_material_graph(package.material_usd_path, recipe.binding)
    assert info.base_color == pytest.approx(recipe.base_color_hint)
    assert info.roughness == pytest.approx(0.42)


def test_shared_authoring_core_preserves_generation_hint_semantics(
    tmp_path: Path,
) -> None:
    recipe = MaterialRecipe(
        id="generation_hint",
        name="Generation Hint",
        description="A display-encoded non-optical generation recipe.",
        appearance_prompt="mid-gray matte polymer",
        base_color_hint=(0.5, 0.5, 0.5),
        pbr_hints=PBRHints(
            roughness=0.42,
            metallic=0.0,
            opacity=0.9,
            transmission=0.5,
        ),
    )
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=recipe,
        material_profile="preview_surface",
        recipe_semantics=MaterialRecipeSemantics.GENERATION_HINTS,
    )

    package = author_material_package(request, tmp_path / "generated")
    info = inspect_material_graph(package.material_usd_path, recipe.binding)

    assert info.base_color == pytest.approx((0.21404114,) * 3)
    assert info.opacity == pytest.approx(0.65)
    manifest = json.loads(package.authoring_manifest_path.read_text(encoding="utf-8"))
    assert manifest["request"]["recipe_semantics"] == "generation_hints"


def test_authoring_request_records_create_provenance_source_only(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    source_material_path = _create_scalar_openpbr(source_usd)
    source = SourceMaterialReference(
        usd_path=source_usd,
        material_prim_path=source_material_path,
    )
    recipe = MaterialRecipe(
        id="re_authored",
        name="Re-authored",
        description="A re-authored material with source provenance.",
        appearance_prompt="re-authored blue coating",
    )
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=recipe,
        provenance_source=source,
        material_profile="preview_surface",
    )

    package = author_material_package(request, tmp_path / "re-authored")
    manifest = json.loads(package.authoring_manifest_path.read_text(encoding="utf-8"))

    assert manifest["request"]["provenance_source"] == source.to_dict()
    with pytest.raises(ValueError, match="provenance_source is create-only"):
        MaterialAuthoringRequest(
            operation="modify",
            recipe=recipe,
            source=source,
            provenance_source=source,
        )
    with pytest.raises(ValueError, match="literal shader values"):
        MaterialAuthoringRequest(
            operation="modify",
            recipe=recipe,
            source=source,
            recipe_semantics="generation_hints",
        )


def test_shared_authoring_core_rejects_tampered_package_manifest(
    tmp_path: Path,
) -> None:
    recipe = MaterialRecipe(
        id="tamper_check",
        name="Tamper Check",
        description="A scalar package used to verify cached package validation.",
        appearance_prompt="smooth blue tamper-check coating",
    )
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=recipe,
        material_profile="preview_surface",
    )
    package_dir = tmp_path / "tampered"
    package = author_material_package(request, package_dir)
    manifest = yaml.safe_load(
        package.materials_manifest_path.read_text(encoding="utf-8")
    )
    manifest["entries"] = []
    package.materials_manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(MaterialPackageAuthoringError, match="validation failed"):
        load_material_package(package_dir)
    with pytest.raises(MaterialPackageAuthoringError, match="validation failed"):
        author_material_package(request, package_dir)


def test_shared_authoring_core_rejects_thin_walled_optics_change(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    source_material_path = _create_scalar_openpbr(source_usd)
    recipe = MaterialRecipe(
        id="invalid_thin_walled",
        name="Invalid Thin Walled",
        description="A request that would change a preserved optical control.",
        appearance_prompt="thin-walled blue coating",
        base_color_hint=(0.2, 0.4, 0.7),
        pbr_hints=PBRHints(roughness=0.25, metallic=1.0, thin_walled=True),
    )

    with pytest.raises(MaterialPackageAuthoringError, match="thin_walled"):
        author_material_package(
            MaterialAuthoringRequest(
                operation="modify",
                recipe=recipe,
                source=SourceMaterialReference(
                    usd_path=source_usd,
                    material_prim_path=source_material_path,
                ),
            ),
            tmp_path / "invalid",
        )


def test_shared_authoring_core_modifies_source_without_mutating_it(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    source_material_path = _create_scalar_openpbr(source_usd, preview_fallback=True)
    source_info = inspect_material_graph(source_usd, source_material_path)
    source_digest = _sha256(source_usd)
    recipe = MaterialRecipe(
        id="refined_openpbr",
        name="Refined OpenPBR",
        description="A warmer, rougher version of the supplied material.",
        appearance_prompt="warm rough blue metal",
        base_color_hint=(0.45, 0.25, 0.15),
        pbr_hints=PBRHints(roughness=0.65, metallic=1.0),
    )
    request = MaterialAuthoringRequest(
        operation="modify",
        recipe=recipe,
        source=SourceMaterialReference(
            usd_path=source_usd,
            material_prim_path=source_material_path,
            usd_sha256=source_digest,
        ),
        target_prim_paths=("/World/Geom/Panel",),
    )

    package = author_material_package(request, tmp_path / "modified")

    assert _sha256(source_usd) == source_digest
    assert package.material_list_entry["source"] == "modified"
    assert package.material_list_entry["target_prim_paths"] == ["/World/Geom/Panel"]
    result = inspect_material_graph(package.material_usd_path, recipe.binding)
    assert result.base_color == pytest.approx(recipe.base_color_hint)
    assert result.roughness == pytest.approx(0.65)
    assert result.metallic == pytest.approx(1.0)
    assert result.material_profile == source_info.material_profile
    assert result.topology_sha256 == source_info.topology_sha256


def test_shared_authoring_core_packages_textured_replacements(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source" / "source.usda"
    source_usd.parent.mkdir()
    source_material_path, _source_textures = _create_textured_openpbr(source_usd)
    replacements = _create_textures(
        tmp_path / "replacements",
        color=(179, 77, 26),
    )
    recipe = MaterialRecipe(
        id="refined_textured_openpbr",
        name="Refined Textured OpenPBR",
        description="A source-preserving material with replacement maps.",
        appearance_prompt="warm textured coating",
    )

    package = author_material_package(
        MaterialAuthoringRequest(
            operation="modify",
            recipe=recipe,
            source=SourceMaterialReference(
                usd_path=source_usd,
                material_prim_path=source_material_path,
            ),
            textures=replacements,
        ),
        tmp_path / "modified",
    )

    published = inspect_material_graph(package.material_usd_path, recipe.binding)
    assert published.representation == "textured_pbr"
    assert published.textures is not None
    assert all(
        path.is_relative_to(package.material_usd_path.parent)
        for path in (
            published.textures.albedo,
            published.textures.normal,
            published.textures.orm,
        )
    )
    assert not (package.material_usd_path.parent / ".authoring_inputs").exists()


def test_shared_authoring_core_rejects_textured_modify_without_replacements(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source" / "source.usda"
    source_usd.parent.mkdir()
    source_material_path, _source_textures = _create_textured_openpbr(source_usd)

    with pytest.raises(
        MaterialPackageAuthoringError,
        match="requires complete replacement textures",
    ):
        author_material_package(
            MaterialAuthoringRequest(
                operation="modify",
                recipe=MaterialRecipe(
                    id="missing_replacements",
                    name="Missing Replacements",
                    description="A textured edit without replacement maps.",
                    appearance_prompt="warmer rough textured coating",
                    base_color_hint=(0.8, 0.2, 0.1),
                    pbr_hints=PBRHints(roughness=0.8, metallic=0.0),
                ),
                source=SourceMaterialReference(
                    usd_path=source_usd,
                    material_prim_path=source_material_path,
                ),
            ),
            tmp_path / "invalid",
        )


def test_shared_authoring_core_binds_manifest_to_returned_material_usd(
    tmp_path: Path,
) -> None:
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=MaterialRecipe(
            id="manifest_binding",
            name="Manifest Binding",
            description="A package used to verify the list manifest target.",
            appearance_prompt="matte blue coating",
        ),
        material_profile="preview_surface",
    )
    package_dir = tmp_path / "package"
    package = author_material_package(request, package_dir)
    alternate = package_dir / "alternate.usda"
    alternate.write_bytes(package.material_usd_path.read_bytes())
    manifest = yaml.safe_load(
        package.materials_manifest_path.read_text(encoding="utf-8")
    )
    manifest["library_path"] = alternate.name
    package.materials_manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(MaterialPackageAuthoringError, match="library_path"):
        load_material_package(package_dir)


def test_shared_authoring_core_rejects_changed_cached_artifact_bytes(
    tmp_path: Path,
) -> None:
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=MaterialRecipe(
            id="artifact_identity",
            name="Artifact Identity",
            description="A package used to verify published artifact hashes.",
            appearance_prompt="matte blue coating",
        ),
        material_profile="preview_surface",
    )
    package_dir = tmp_path / "package"
    package = author_material_package(request, package_dir)
    package.material_usd_path.write_bytes(
        package.material_usd_path.read_bytes() + b"\n# changed after publication\n"
    )

    with pytest.raises(MaterialPackageAuthoringError, match="artifact digest"):
        load_material_package(package_dir)
    with pytest.raises(MaterialPackageAuthoringError, match="artifact digest"):
        author_material_package(request, package_dir)


def test_shared_authoring_core_freezes_and_verifies_texture_identity(
    tmp_path: Path,
) -> None:
    textures = _create_textures(tmp_path / "textures", color=(64, 96, 128))
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=MaterialRecipe(
            id="frozen_textures",
            name="Frozen Textures",
            description="A textured request with frozen input identity.",
            appearance_prompt="matte blue woven polymer",
        ),
        textures=textures,
        material_profile="preview_surface",
    )
    frozen_request_id = request.request_id
    frozen_digest = request.to_dict()["textures"]["albedo"]["sha256"]

    Image.new("RGB", (16, 16), (255, 0, 0)).save(textures.albedo)

    assert request.request_id == frozen_request_id
    assert request.to_dict()["textures"]["albedo"]["sha256"] == frozen_digest
    with pytest.raises(MaterialPackageAuthoringError, match="frozen digest"):
        author_material_package(request, tmp_path / "changed")


def test_shared_authoring_core_rejects_16_bit_texture_before_copy(
    tmp_path: Path,
) -> None:
    textures = _create_textures(tmp_path / "textures", color=(64, 96, 128))
    Image.new("I;16", (16, 16), 32768).save(textures.albedo)
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=MaterialRecipe(
            id="high_bit_depth",
            name="High Bit Depth",
            description="A material with an unsupported high-bit-depth map.",
            appearance_prompt="matte gray polymer",
        ),
        textures=textures,
        material_profile="preview_surface",
    )

    with pytest.raises(MaterialPackageAuthoringError, match="8-bit channels"):
        author_material_package(request, tmp_path / "invalid")


def test_shared_authoring_core_preserves_albedo_alpha_when_packaging(
    tmp_path: Path,
) -> None:
    textures = _create_textures(tmp_path / "textures", color=(64, 96, 128))
    Image.new("RGBA", (16, 16), (64, 96, 128, 37)).save(textures.albedo)
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=MaterialRecipe(
            id="rgba_albedo",
            name="RGBA Albedo",
            description="A material whose albedo carries visible alpha.",
            appearance_prompt="translucent blue polymer",
        ),
        textures=textures,
        material_profile="preview_surface",
    )

    package = author_material_package(request, tmp_path / "rgba")

    assert package.textures is not None
    with Image.open(package.textures.albedo) as image:
        assert image.mode == "RGBA"
        assert image.getchannel("A").getextrema() == (37, 37)


def test_shared_authoring_core_rejects_temporary_texture_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.material_library_generation.authoring as authoring_module

    source_usd = tmp_path / "source" / "source.usda"
    source_usd.parent.mkdir()
    source_material_path, _source_textures = _create_textured_openpbr(source_usd)
    replacements = _create_textures(tmp_path / "replacements", color=(179, 77, 26))
    recipe = MaterialRecipe(
        id="invalid_temporary_reference",
        name="Invalid Temporary Reference",
        description="A package whose writer reports a staged texture reference.",
        appearance_prompt="warm textured coating",
    )
    original_write = authoring_module.write_edited_material_graphs

    def write_with_temporary_references(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[MaterialGraphInfo, ...]:
        (published,) = original_write(*args, **kwargs)
        library_path = Path(args[0]).resolve()
        staged_dir = (
            library_path.parent / ".authoring_inputs" / "textures" / recipe.material_id
        )
        return (
            replace(
                published,
                textures=TextureMapSet(
                    albedo=staged_dir / "albedo.png",
                    normal=staged_dir / "normal.png",
                    orm=staged_dir / "orm.png",
                ),
            ),
        )

    monkeypatch.setattr(
        authoring_module,
        "write_edited_material_graphs",
        write_with_temporary_references,
    )

    with pytest.raises(
        MaterialPackageAuthoringError,
        match="temporary authoring input texture",
    ):
        author_material_package(
            MaterialAuthoringRequest(
                operation="modify",
                recipe=recipe,
                source=SourceMaterialReference(
                    usd_path=source_usd,
                    material_prim_path=source_material_path,
                ),
                textures=replacements,
            ),
            tmp_path / "invalid",
        )


def test_shared_authoring_core_detects_source_change_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.material_library_generation.authoring as authoring_module

    source_usd = tmp_path / "source.usda"
    source_material_path = _create_scalar_preview_surface(source_usd)
    recipe = MaterialRecipe(
        id="concurrent_change",
        name="Concurrent Change",
        description="A source-backed request used to exercise digest checks.",
        appearance_prompt="rough blue coating",
    )
    request = MaterialAuthoringRequest(
        operation="modify",
        recipe=recipe,
        source=SourceMaterialReference(
            usd_path=source_usd,
            material_prim_path=source_material_path,
            usd_sha256=_sha256(source_usd),
        ),
    )
    original_write = authoring_module.write_edited_material_graphs

    def write_then_change_source(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[MaterialGraphInfo, ...]:
        result = original_write(*args, **kwargs)
        source_usd.write_bytes(source_usd.read_bytes() + b"\n# concurrent change\n")
        return result

    monkeypatch.setattr(
        authoring_module,
        "write_edited_material_graphs",
        write_then_change_source,
    )

    with pytest.raises(MaterialPackageAuthoringError, match="changed while"):
        author_material_package(request, tmp_path / "modified")


def test_shared_authoring_core_rejects_source_changed_after_request_freeze(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    source_material_path = _create_scalar_preview_surface(source_usd)
    request = MaterialAuthoringRequest(
        operation="modify",
        recipe=MaterialRecipe(
            id="frozen_source",
            name="Frozen Source",
            description="A source-backed request with frozen identity.",
            appearance_prompt="rough blue coating",
        ),
        source=SourceMaterialReference(
            usd_path=source_usd,
            material_prim_path=source_material_path,
        ),
    )
    frozen_request_id = request.request_id

    source_usd.write_bytes(source_usd.read_bytes() + b"\n# replaced after admission\n")

    assert request.request_id == frozen_request_id
    with pytest.raises(MaterialPackageAuthoringError, match="frozen digest"):
        author_material_package(request, tmp_path / "modified")


def test_shared_authoring_core_rejects_nonportable_validation_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import material_agent.material_library_generation.authoring as authoring_module

    monkeypatch.setattr(
        authoring_module,
        "validate_generated_material_library",
        lambda _path: SimpleNamespace(
            ok=True,
            errors=(),
            warnings=(
                "non-relative asset path in material library: https://example.test/map.png",
            ),
            metadata={},
        ),
    )
    request = MaterialAuthoringRequest(
        operation="create",
        recipe=MaterialRecipe(
            id="nonportable",
            name="Nonportable",
            description="A package whose validation reports a remote asset.",
            appearance_prompt="matte blue polymer",
        ),
        material_profile="preview_surface",
    )

    with pytest.raises(MaterialPackageAuthoringError, match="not portable"):
        author_material_package(request, tmp_path / "nonportable")


def test_shared_authoring_core_modifies_scalar_omnipbr_package(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    source_material_path = _create_scalar_omnipbr_mdl(source_usd)
    source_info = inspect_material_graph(source_usd, source_material_path)
    recipe = MaterialRecipe(
        id="refined_omnipbr",
        name="Refined OmniPBR",
        description="A warmer scalar OmniPBR material.",
        appearance_prompt="warm brushed metal",
        base_color_hint=(0.35, 0.18, 0.08),
        pbr_hints=PBRHints(roughness=0.58, metallic=1.0),
    )

    package = author_material_package(
        MaterialAuthoringRequest(
            operation="modify",
            recipe=recipe,
            source=SourceMaterialReference(
                usd_path=source_usd,
                material_prim_path=source_material_path,
            ),
        ),
        tmp_path / "modified",
    )

    result = inspect_material_graph(package.material_usd_path, recipe.binding)
    assert result.material_profile == "omnipbr_mdl"
    assert result.representation == "scalar_pbr"
    assert result.shader_ids == source_info.shader_ids
    assert result.topology_sha256 == source_info.topology_sha256
    assert package.validation["texture_count"] == 0


def test_shared_authoring_core_rejects_representation_change(
    tmp_path: Path,
) -> None:
    source_usd = tmp_path / "source.usda"
    source_material_path = _create_scalar_preview_surface(source_usd)
    textures = _create_textures(tmp_path / "replacement", color=(10, 20, 30))
    recipe = MaterialRecipe(
        name="Invalid textured replacement",
        description="Must not change a scalar source into a textured graph.",
        appearance_prompt="textured replacement",
    )

    with pytest.raises(MaterialPackageAuthoringError, match="forbids adding textures"):
        author_material_package(
            MaterialAuthoringRequest(
                operation="modify",
                recipe=recipe,
                source=SourceMaterialReference(
                    usd_path=source_usd,
                    material_prim_path=source_material_path,
                ),
                textures=textures,
            ),
            tmp_path / "invalid",
        )
