# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused branch coverage for USD material helpers."""

from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image

pxr = pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

from world_understanding.utils.usd import material as material_utils  # noqa: E402


class _RaisingConnectedSources:
    def GetConnectedSources(self) -> tuple[list[object], None]:
        raise RuntimeError("connection lookup failed")


class _ConnectedSources:
    def __init__(self, sources: list[object]) -> None:
        self._sources = sources

    def GetConnectedSources(self) -> tuple[list[object], None]:
        return self._sources, None


class _SourceInfo:
    def __init__(self, source: object | None, source_name: str = "result") -> None:
        self.source = source
        self.sourceName = source_name


class _MaterialWithOutputs:
    def __init__(self, outputs: dict[str, object | None]) -> None:
        self._outputs = outputs

    def GetSurfaceOutput(self, render_context: str = "") -> object | None:
        return self._outputs.get(render_context)


def _create_openpbr_material(
    stage: Usd.Stage,
    material_path: str = "/World/Looks/Gold",
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/OpenPBR")
    shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )
    material.CreateSurfaceOutput()
    return material


def _define_triangle_mesh(
    stage: Usd.Stage,
    path: str,
    *,
    with_st: bool = True,
    interpolation: str = "faceVarying",
    uv_values: object | None = None,
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ],
    )
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    if with_st:
        st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            interpolation,
        )
        st.Set(
            uv_values
            if uv_values is not None
            else Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.0, 0.0),
                    Gf.Vec2f(1.0, 0.0),
                    Gf.Vec2f(0.0, 1.0),
                ],
            ),
        )
    return mesh


def _make_texture(path: Path, color: tuple[int, int, int] = (8, 80, 160)) -> Path:
    Image.new("RGB", (2, 2), color).save(path)
    return path


def test_connection_helpers_handle_empty_outputs_and_errors() -> None:
    class _NonShaderPrim:
        def IsA(self, schema: object) -> bool:
            return False

    class _Source:
        def __init__(self, prim: object) -> None:
            self._prim = prim

        def GetPrim(self) -> object:
            return self._prim

    assert not material_utils._output_has_connected_source(_RaisingConnectedSources())
    assert not material_utils._material_has_connected_surface(
        _MaterialWithOutputs({"": None}),
    )
    assert not material_utils._material_has_connected_texture_capable_mdl_surface(
        _MaterialWithOutputs({"mdl": None}),
    )
    assert not material_utils._material_has_connected_texture_capable_mdl_surface(
        _MaterialWithOutputs({"mdl": _RaisingConnectedSources()}),
    )
    assert not material_utils._material_has_connected_texture_capable_mdl_surface(
        _MaterialWithOutputs({"mdl": _ConnectedSources([_SourceInfo(None)])}),
    )
    assert not material_utils._material_has_connected_texture_capable_mdl_surface(
        _MaterialWithOutputs(
            {"mdl": _ConnectedSources([_SourceInfo(_Source(_NonShaderPrim()))])},
        ),
    )

    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/World/NotShader")
    assert not material_utils._connected_materialx_openpbr_surface(
        _MaterialWithOutputs({"mtlx": None}),
    )
    assert not material_utils._connected_materialx_openpbr_surface(
        _MaterialWithOutputs({"mtlx": _RaisingConnectedSources()}),
    )
    assert not material_utils._connected_materialx_openpbr_surface(
        _MaterialWithOutputs({"mtlx": _ConnectedSources([_SourceInfo(None)])}),
    )
    assert not material_utils._connected_materialx_openpbr_surface(
        _MaterialWithOutputs(
            {"mtlx": _ConnectedSources([_SourceInfo(_Source(xform.GetPrim()))])},
        ),
    )
    assert not material_utils._material_surface_is_ovrtx_preview_fallback(
        _MaterialWithOutputs({"": None}),
    )
    assert not material_utils._material_surface_is_ovrtx_preview_fallback(
        _MaterialWithOutputs({"": _RaisingConnectedSources()}),
    )


def test_mdl_texture_capable_surface_detects_values_connections_and_cycles() -> None:
    class _FakePrim:
        def __init__(self, path: str = "/FakeShader", is_shader: bool = True) -> None:
            self._path = path
            self._is_shader = is_shader

        def GetPath(self) -> str:
            return self._path

        def IsA(self, schema: object) -> bool:
            return self._is_shader and schema is UsdShade.Shader

    class _FakeSource:
        def __init__(self, prim: _FakePrim) -> None:
            self._prim = prim

        def GetPrim(self) -> _FakePrim:
            return self._prim

    class _FakeInput:
        def __init__(
            self,
            *,
            sources: list[object] | None = None,
            raises: bool = False,
        ) -> None:
            self._sources = sources or []
            self._raises = raises

        def GetBaseName(self) -> str:
            return "passthrough"

        def GetConnectedSources(self) -> tuple[list[object], None]:
            if self._raises:
                raise RuntimeError("failed")
            return self._sources, None

    class _FakeShader:
        def __init__(self, inputs: list[_FakeInput]) -> None:
            self._inputs = inputs

        def GetPrim(self) -> _FakePrim:
            return _FakePrim()

        def GetInputs(self) -> list[_FakeInput]:
            return self._inputs

    assert not material_utils._shader_has_texture_capable_input(
        _FakeShader([_FakeInput(raises=True)]),
        visited=set(),
    )
    assert not material_utils._shader_has_texture_capable_input(
        _FakeShader([_FakeInput(sources=[_SourceInfo(None)])]),
        visited=set(),
    )
    assert not material_utils._shader_has_texture_capable_input(
        _FakeShader(
            [
                _FakeInput(
                    sources=[_SourceInfo(_FakeSource(_FakePrim(is_shader=False)))]
                )
            ],
        ),
        visited=set(),
    )

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Mdl")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("albedo.png"),
    )
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )

    assert material_utils._material_has_connected_texture_capable_mdl_surface(material)
    assert not material_utils._shader_has_texture_capable_input(
        shader,
        visited={str(shader.GetPath())},
    )

    string_shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/StringMdl")
    string_shader.CreateInput("normal_texture", Sdf.ValueTypeNames.String).Set(
        "normal.png",
    )
    assert material_utils._shader_has_texture_capable_input(
        string_shader,
        visited=set(),
    )

    connected_shader = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Painted/ConnectedMdl",
    )
    upstream = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Upstream")
    connected_shader.CreateInput(
        "roughness_texture",
        Sdf.ValueTypeNames.Asset,
    ).ConnectToSource(upstream.CreateOutput("out", Sdf.ValueTypeNames.Token))
    assert material_utils._shader_has_texture_capable_input(
        connected_shader,
        visited=set(),
    )

    parent = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Parent")
    child = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Child")
    child.CreateInput("metallic_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("metal.png"),
    )
    parent.CreateInput("not_texture", Sdf.ValueTypeNames.Float).ConnectToSource(
        child.CreateOutput("result", Sdf.ValueTypeNames.Float),
    )
    assert material_utils._shader_has_texture_capable_input(parent, visited=set())

    plain_shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Plain")
    plain_shader.CreateInput("color", Sdf.ValueTypeNames.Color3f).Set((1, 1, 1))
    assert not material_utils._shader_has_texture_capable_input(
        plain_shader,
        visited=set(),
    )


def test_material_texture_asset_helpers_walk_mdl_and_preview_networks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeInput:
        def __init__(
            self,
            *,
            sources: list[object] | None = None,
            raises: bool = False,
        ) -> None:
            self._sources = sources or []
            self._raises = raises

        def GetConnectedSources(self) -> tuple[list[object], None]:
            if self._raises:
                raise RuntimeError("failed")
            return self._sources, None

    class _FakeShader:
        def __init__(self, inp: object | None) -> None:
            self._input = inp

        def GetInput(self, name: str) -> object | None:
            return self._input

    assert (
        material_utils._input_connected_shader_asset(_FakeShader(None), "missing")
        is None
    )
    assert (
        material_utils._input_connected_shader_asset(
            _FakeShader(_FakeInput(raises=True)),
            "file",
        )
        is None
    )
    assert (
        material_utils._input_connected_shader_asset(
            _FakeShader(_FakeInput(sources=[_SourceInfo(None)])),
            "file",
        )
        is None
    )

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    UsdGeom.Xform.Define(stage, "/World/Looks/Painted/Group")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Mdl")

    assert (
        material_utils._mdl_shader_input_asset(
            material.GetPrim(),
            "diffuse_texture",
        )
        is None
    )
    assert (
        material_utils._mdl_shader_input_value(material.GetPrim(), "diffuse_tint")
        is None
    )
    assert material_utils._shader_input_asset(shader, "missing") is None
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("diffuse.png"),
    )
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
        (0.1, 0.2, 0.3),
    )
    shader.CreateInput("label", Sdf.ValueTypeNames.String).Set("not-an-asset")

    assert (
        material_utils._mdl_shader_input_asset(
            material.GetPrim(),
            "diffuse_texture",
        ).path
        == "diffuse.png"
    )
    assert material_utils._mdl_shader_input_value(
        material.GetPrim(),
        "diffuse_tint",
    ) == pytest.approx((0.1, 0.2, 0.3))
    assert material_utils._shader_input_asset(shader, "label") is None

    unconnected_material = UsdShade.Material.Define(stage, "/World/Looks/Unconnected")
    assert (
        material_utils._connected_preview_diffuse_texture_asset(
            unconnected_material.GetPrim(),
        )
        is None
    )

    class _FakeMaterial:
        def __init__(self, prim: object) -> None:
            self.prim = prim

        def GetSurfaceOutput(self) -> object | None:
            return None

    class _FakeRaisingMaterial:
        def __init__(self, prim: object) -> None:
            self.prim = prim

        def GetSurfaceOutput(self) -> _RaisingConnectedSources:
            return _RaisingConnectedSources()

    class _FakeSourceNoneMaterial:
        def __init__(self, prim: object) -> None:
            self.prim = prim

        def GetSurfaceOutput(self) -> _ConnectedSources:
            return _ConnectedSources([_SourceInfo(None)])

    with monkeypatch.context() as patch:
        patch.setattr(material_utils.UsdShade, "Material", _FakeMaterial)
        assert material_utils._connected_preview_diffuse_texture_asset(object()) is None
    with monkeypatch.context() as patch:
        patch.setattr(material_utils.UsdShade, "Material", _FakeRaisingMaterial)
        assert material_utils._connected_preview_diffuse_texture_asset(object()) is None
    with monkeypatch.context() as patch:
        patch.setattr(material_utils.UsdShade, "Material", _FakeSourceNoneMaterial)
        assert material_utils._connected_preview_diffuse_texture_asset(object()) is None

    preview = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Preview")
    texture = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("preview.png"),
    )
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3),
    )
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )

    assert (
        material_utils._connected_preview_diffuse_texture_asset(material.GetPrim()).path
        == "preview.png"
    )
    assert (
        material_utils._preview_base_color_texture_asset(material.GetPrim()).path
        == "diffuse.png"
    )

    recursive_material = UsdShade.Material.Define(stage, "/World/Looks/Recursive")
    recursive_preview = UsdShade.Shader.Define(stage, "/World/Looks/Recursive/Preview")
    intermediate = UsdShade.Shader.Define(stage, "/World/Looks/Recursive/Mixer")
    recursive_texture = UsdShade.Shader.Define(
        stage,
        "/World/Looks/Recursive/Texture",
    )
    recursive_texture.CreateIdAttr("UsdUVTexture")
    recursive_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("recursive.png"),
    )
    intermediate.CreateInput("result", Sdf.ValueTypeNames.Float3).ConnectToSource(
        recursive_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3),
    )
    recursive_preview.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).ConnectToSource(intermediate.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    recursive_material.CreateSurfaceOutput().ConnectToSource(
        recursive_preview.CreateOutput("surface", Sdf.ValueTypeNames.Token),
    )

    assert (
        material_utils._connected_preview_diffuse_texture_asset(
            recursive_material.GetPrim(),
        ).path
        == "recursive.png"
    )
    assert (
        material_utils._input_connected_shader_asset(
            intermediate,
            "result",
            visited={str(recursive_texture.GetPath())},
        )
        is None
    )


def test_color_sampling_and_primvar_helpers_cover_fallback_paths(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Color")

    assert material_utils._openpbr_preview_diffuse_color(
        material.GetPrim(),
        0.75,
    ) == pytest.approx((0.8, 0.8, 0.8))
    assert (
        material_utils._float_material_input(material.GetPrim(), "missing", 0.4) == 0.4
    )
    material.GetPrim().CreateAttribute(
        "inputs:unset_float",
        Sdf.ValueTypeNames.Float,
    )
    assert (
        material_utils._float_material_input(material.GetPrim(), "unset_float", 0.6)
        == 0.6
    )
    material.GetPrim().CreateAttribute(
        "inputs:bad_float", Sdf.ValueTypeNames.String
    ).Set(
        "not-float",
    )
    assert (
        material_utils._float_material_input(material.GetPrim(), "bad_float", 0.25)
        == 0.25
    )

    assert material_utils._coerce_color3f(None) is None
    assert material_utils._coerce_color3f(object()) is None
    assert material_utils._coerce_color3f((1.0, 2.0)) is None
    assert material_utils._coerce_color3f(("bad", 0.0, 1.0)) is None
    assert material_utils._coerce_color3f((-1.0, 0.5, 2.0, 99.0)) == pytest.approx(
        (0.0, 0.5, 1.0),
    )

    shader = UsdShade.Shader.Define(stage, "/World/Looks/Color/Mdl")
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
        (0.2, 0.4, 0.6),
    )
    assert material_utils._preview_base_color_value(
        material.GetPrim(),
        Sdf.AssetPath("missing.png"),
        base_dir=tmp_path,
    ) == pytest.approx((0.2, 0.4, 0.6))

    class _ZeroImage:
        size = (0, 4)

        def getpixel(self, xy: object) -> tuple[int, int, int]:
            return (0, 0, 0)

    class _ThrowingImage:
        size = (2, 2)

        def getpixel(self, xy: object) -> tuple[int, int, int]:
            raise RuntimeError("bad pixel")

    assert material_utils._sample_texture_at_uv(_ZeroImage(), (0.5, 0.5)) is None
    assert material_utils._sample_texture_at_uv(_ThrowingImage(), (0.5, 0.5)) is None
    assert material_utils._sample_texture_at_uv(_ThrowingImage(), object()) is None

    invalid_image = tmp_path / "not-image.png"
    invalid_image.write_text("not an image", encoding="utf-8")
    assert (
        material_utils._sample_texture_average_color(
            Sdf.AssetPath(str(invalid_image)),
        )
        is None
    )
    assert material_utils._open_texture_image(Sdf.AssetPath(str(invalid_image))) is None

    assert material_utils._expanded_primvar_values(1, [], 1) is None
    assert material_utils._expanded_primvar_values(["a"], [], 0) == []
    assert material_utils._expanded_primvar_values(["a"], object(), 1) == ["a"]
    assert material_utils._expanded_primvar_values(["a", "b"], [1, 0], 2) == [
        "b",
        "a",
    ]
    assert material_utils._expanded_primvar_values(["a"], [3], 1) is None
    assert material_utils._expanded_primvar_values(["a", "b"], [], 2) == ["a", "b"]
    assert material_utils._expanded_primvar_values(["a"], [], 2) is None


def test_mesh_uv_helpers_and_display_color_bake_skip_unusable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    texture_path = _make_texture(tmp_path / "albedo.png")
    textured = UsdShade.Material.Define(stage, "/World/Looks/Textured")
    textured.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    plain = UsdShade.Material.Define(stage, "/World/Looks/Plain")

    bake_empty = material_utils.bake_texture_file_materials_to_display_color_for_render(
        Usd.Stage.CreateInMemory(),
    )
    assert bake_empty == 0

    no_st = _define_triangle_mesh(stage, "/World/NoSt", with_st=False)
    assert material_utils._mesh_uv_values_for_display_color(no_st) is None
    UsdShade.MaterialBindingAPI.Apply(no_st.GetPrim()).Bind(textured)

    class _FakePrimvar:
        def __init__(self, value: object = [(0.0, 0.0)]) -> None:
            self._value = value

        def HasValue(self) -> bool:
            return True

        def Get(self) -> object:
            return self._value

        def GetInterpolation(self) -> str:
            return "custom"

        def GetIndices(self) -> list[int]:
            return []

    class _FakePrimvarsAPI:
        value: object = [(0.0, 0.0)]

        def __init__(self, prim: object) -> None:
            self.prim = prim

        def GetPrimvar(self, name: str) -> _FakePrimvar:
            return _FakePrimvar(self.value)

    class _FakeAttr:
        def __init__(self, value: object) -> None:
            self._value = value

        def Get(self) -> object:
            return self._value

    class _FakeMesh:
        def GetPrim(self) -> object:
            return object()

        def GetFaceVertexCountsAttr(self) -> _FakeAttr:
            return _FakeAttr([3])

        def GetPointsAttr(self) -> _FakeAttr:
            return _FakeAttr([object(), object(), object()])

    with monkeypatch.context() as patch:
        patch.setattr(material_utils.UsdGeom, "PrimvarsAPI", _FakePrimvarsAPI)
        _FakePrimvarsAPI.value = None
        assert material_utils._mesh_uv_values_for_display_color(_FakeMesh()) is None
        _FakePrimvarsAPI.value = [(0.0, 0.0)]
        assert material_utils._mesh_uv_values_for_display_color(_FakeMesh()) is None

    mismatch = _define_triangle_mesh(
        stage,
        "/World/Mismatch",
        uv_values=Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0)]),
    )
    assert material_utils._mesh_uv_values_for_display_color(mismatch) is None
    UsdShade.MaterialBindingAPI.Apply(mismatch.GetPrim()).Bind(textured)

    instanceable_mesh = _define_triangle_mesh(stage, "/World/Instanceable")
    instanceable_mesh.GetPrim().SetInstanceable(True)
    UsdShade.MaterialBindingAPI.Apply(instanceable_mesh.GetPrim()).Bind(textured)

    unbound = _define_triangle_mesh(stage, "/World/Unbound")
    assert unbound
    bound_to_plain = _define_triangle_mesh(stage, "/World/PlainBound")
    UsdShade.MaterialBindingAPI.Apply(bound_to_plain.GetPrim()).Bind(plain)

    missing_texture_material = UsdShade.Material.Define(stage, "/World/Looks/Missing")
    missing_texture_material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(tmp_path / "missing.png")))
    missing_texture_mesh = _define_triangle_mesh(stage, "/World/MissingTexture")
    UsdShade.MaterialBindingAPI.Apply(missing_texture_mesh.GetPrim()).Bind(
        missing_texture_material,
    )

    assert (
        material_utils.bake_texture_file_materials_to_display_color_for_render(stage)
        == 1
    )
    assert not instanceable_mesh.GetPrim().IsInstanceable()
    display_color = UsdGeom.PrimvarsAPI(instanceable_mesh.GetPrim()).GetPrimvar(
        "displayColor",
    )
    assert display_color.HasValue()

    class _FakeRootLayer:
        realPath = ""

    class _FakePrim:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def IsA(self, schema: object) -> bool:
            return (self.kind == "material" and schema is UsdShade.Material) or (
                self.kind == "mesh" and schema is UsdGeom.Mesh
            )

        def IsInstanceProxy(self) -> bool:
            return self.kind == "mesh"

        def GetPath(self) -> str:
            return f"/Fake/{self.kind}"

    class _FakeStage:
        def GetRootLayer(self) -> _FakeRootLayer:
            return _FakeRootLayer()

        def Traverse(self) -> list[_FakePrim]:
            return [_FakePrim("material"), _FakePrim("mesh")]

    with monkeypatch.context() as patch:
        patch.setattr(
            material_utils,
            "_preview_base_color_texture_asset",
            lambda prim: Sdf.AssetPath("albedo.png")
            if prim.kind == "material"
            else None,
        )
        assert (
            material_utils.bake_texture_file_materials_to_display_color_for_render(
                _FakeStage(),
            )
            == 0
        )


def test_preview_fallback_authoring_guard_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InvalidMaterial:
        def GetPrim(self) -> None:
            return None

    class _ProxyPrim:
        def IsValid(self) -> bool:
            return True

        def IsInstanceProxy(self) -> bool:
            return True

        def IsInstance(self) -> bool:
            return False

        def IsInstanceable(self) -> bool:
            return False

    class _ProxyMaterial:
        def GetPrim(self) -> _ProxyPrim:
            return _ProxyPrim()

    assert not material_utils._prepare_material_for_surface_authoring(
        _InvalidMaterial(),
    )
    assert not material_utils._prepare_material_for_surface_authoring(_ProxyMaterial())

    stage = Usd.Stage.CreateInMemory()
    source = UsdShade.Material.Define(stage, "/World/Looks/Source")
    source.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(_make_texture(tmp_path / "albedo.png"))))

    class _InstanceProxyPrim:
        def IsValid(self) -> bool:
            return True

        def IsInstanceProxy(self) -> bool:
            return True

        def IsInstance(self) -> bool:
            return False

        def IsInstanceable(self) -> bool:
            return False

    class _EarlyReturnStage:
        def GetPrimAtPath(self, path: str) -> _InstanceProxyPrim:
            return _InstanceProxyPrim()

    assert not material_utils._author_ovrtx_preview_fallback(
        _EarlyReturnStage(),
        "/World/Looks/Source",
        source.GetPrim(),
    )
    assert not material_utils._author_ovrtx_textured_preview_fallback(
        _EarlyReturnStage(),
        "/World/Looks/Source",
        source.GetPrim(),
        Sdf.AssetPath(str(tmp_path / "albedo.png")),
    )

    instance_source = UsdShade.Material.Define(stage, "/World/Looks/InstanceSource")
    instance_source.GetPrim().SetInstanceable(True)
    assert material_utils._author_ovrtx_textured_preview_fallback(
        stage,
        "/World/Looks/InstanceSource",
        instance_source.GetPrim(),
        Sdf.AssetPath(str(tmp_path / "albedo.png")),
    )
    assert not instance_source.GetPrim().IsInstanceable()

    monkeypatch.setattr(
        material_utils,
        "_prepare_material_for_surface_authoring",
        lambda material: False,
    )
    assert not material_utils._suppress_materialx_surface(source)
    assert not material_utils._author_ovrtx_preview_fallback(
        stage,
        "/World/Looks/BlockedPreview",
        source.GetPrim(),
    )
    assert not material_utils._author_ovrtx_textured_preview_fallback(
        stage,
        "/World/Looks/BlockedTextured",
        source.GetPrim(),
        Sdf.AssetPath(str(tmp_path / "albedo.png")),
    )


def test_textured_preview_fallback_skips_connected_texture_capable_mdl_surface() -> (
    None
):
    empty_stage = Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(empty_stage, "/World/Looks/NoTexture")
    assert (
        material_utils.add_ovrtx_preview_fallbacks_for_texture_file_materials(
            empty_stage,
        )
        == 0
    )

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("generated.png"))
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/Mdl")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("mdl.png"),
    )
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )

    assert (
        material_utils.add_ovrtx_preview_fallbacks_for_texture_file_materials(
            stage,
            skip_connected_mdl_surface=True,
        )
        == 1
    )

    generated_stage = Usd.Stage.CreateInMemory()
    generated_material = UsdShade.Material.Define(
        generated_stage,
        "/World/Looks/Painted",
    )
    generated_material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("generated.png"))
    generated_shader = UsdShade.Shader.Define(
        generated_stage,
        "/World/Looks/Painted/Mdl",
    )
    generated_shader.CreateInput(
        "diffuse_texture",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("generated.png"))
    generated_material.CreateSurfaceOutput("mdl").ConnectToSource(
        generated_shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
    )

    assert (
        material_utils.add_ovrtx_preview_fallbacks_for_texture_file_materials(
            generated_stage,
            skip_connected_mdl_surface=True,
        )
        == 0
    )


def test_textured_preview_fallback_asset_matching_helpers_cover_edges() -> None:
    assert (
        material_utils._normalized_asset_path_key(" https://example.com/a%20b.png ")
        == "https://example.com/a b.png"
    )
    assert material_utils._asset_path_keys(r"textures\albedo.png") == {
        "textures/albedo.png"
    }

    stage = Usd.Stage.CreateInMemory()
    uv_texture = UsdShade.Shader.Define(stage, "/World/Looks/UVTexture")
    uv_texture.CreateIdAttr("UsdUVTexture")
    uv_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.png"),
    )
    assert material_utils._shader_uses_texture_asset(
        uv_texture,
        Sdf.AssetPath("textures/albedo.png"),
        visited=set(),
    )
    assert not material_utils._shader_uses_texture_asset(
        uv_texture,
        Sdf.AssetPath("textures/albedo.png"),
        visited={str(uv_texture.GetPath())},
    )

    mdl_shader = UsdShade.Shader.Define(stage, "/World/Looks/Mdl")
    mdl_shader.CreateInput("tint", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        uv_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3),
    )
    assert material_utils._shader_uses_texture_asset(
        mdl_shader,
        Sdf.AssetPath("textures/albedo.png"),
        visited=set(),
    )


def test_textured_preview_fallback_asset_matching_defensive_branches() -> None:
    class _Prim:
        def __init__(self, path: str, is_shader: bool = True) -> None:
            self._path = path
            self._is_shader = is_shader

        def GetPath(self) -> str:
            return self._path

        def IsA(self, _schema: object) -> bool:
            return self._is_shader

    class _Source:
        def __init__(self, prim: _Prim | None = None) -> None:
            self._prim = prim

        def GetPrim(self) -> _Prim | None:
            return self._prim

    class _SourceInfo:
        def __init__(self, source: object | None) -> None:
            self.source = source

    class _Input:
        def __init__(
            self,
            *,
            source: object | None = None,
            raises: bool = False,
        ) -> None:
            self._source = source
            self._raises = raises

        def GetBaseName(self) -> str:
            return "not_a_texture"

        def GetConnectedSources(self) -> tuple[list[_SourceInfo], list[object]]:
            if self._raises:
                raise RuntimeError("bad connection")
            return [_SourceInfo(self._source)], []

    class _Shader:
        def __init__(self, inputs: list[_Input]) -> None:
            self._inputs = inputs

        def GetPrim(self) -> _Prim:
            return _Prim("/World/Looks/FakeShader")

        def GetIdAttr(self) -> None:
            return None

        def GetInputs(self) -> list[_Input]:
            return self._inputs

    assert not material_utils._shader_uses_texture_asset(
        _Shader(
            [
                _Input(raises=True),
                _Input(source=None),
                _Input(source=_Source(_Prim("/World/Looks/NotShader", False))),
            ]
        ),
        Sdf.AssetPath("textures/albedo.png"),
        visited=set(),
    )

    class _Output:
        def __init__(
            self,
            *,
            sources: list[_SourceInfo] | None = None,
            raises: bool = False,
        ) -> None:
            self._sources = sources or []
            self._raises = raises

        def GetConnectedSources(self) -> tuple[list[_SourceInfo], list[object]]:
            if self._raises:
                raise RuntimeError("bad output")
            return self._sources, []

    class _Material:
        def __init__(self, output: _Output) -> None:
            self._output = output

        def GetSurfaceOutput(self, render_context: str = "") -> _Output:
            assert render_context == "mdl"
            return self._output

    assert not material_utils._connected_mdl_surface_uses_texture_asset(
        _Material(_Output(raises=True)),
        Sdf.AssetPath("textures/albedo.png"),
    )
    assert not material_utils._connected_mdl_surface_uses_texture_asset(
        _Material(
            _Output(
                sources=[
                    _SourceInfo(None),
                    _SourceInfo(_Source(_Prim("/World/Looks/NotShader", False))),
                ],
            )
        ),
        Sdf.AssetPath("textures/albedo.png"),
    )


def test_materialx_and_stage_file_fallback_helpers_cover_empty_and_skip_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(stage, "/World/Looks/NoOpenPBR")
    assert material_utils._iter_materialx_openpbr_fallback_prims(stage) == []

    material = _create_openpbr_material(stage)

    monkeypatch.setattr(
        material_utils,
        "_iter_materialx_openpbr_fallback_prims",
        lambda stage_arg, target_paths=None: [material.GetPrim()],
    )
    monkeypatch.setattr(
        material_utils,
        "_iter_materialx_openpbr_surface_prims",
        lambda stage_arg, target_paths=None: [material.GetPrim()],
    )
    monkeypatch.setattr(
        material_utils,
        "_author_ovrtx_preview_fallback",
        lambda *args, **kwargs: True,
    )
    assert (
        material_utils.add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 1
    )

    plain_material = UsdShade.Material.Define(stage, "/World/Looks/Plain")
    monkeypatch.setattr(
        material_utils,
        "_iter_materialx_openpbr_fallback_prims",
        lambda stage_arg, target_paths=None: [],
    )
    monkeypatch.setattr(
        material_utils,
        "_iter_materialx_openpbr_surface_prims",
        lambda stage_arg, target_paths=None: [plain_material.GetPrim()],
    )
    assert (
        material_utils.add_ovrtx_preview_fallbacks_for_materialx_openpbr(
            stage,
            suppress_materialx_surface=True,
        )
        == 0
    )
    assert (
        material_utils.write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
            stage,
            "unused.usda",
        )
        == 0
    )

    class _FakeStageType:
        @staticmethod
        def Open(path: str) -> None:
            return None

    monkeypatch.setattr(material_utils.Usd, "Stage", _FakeStageType)
    assert material_utils.add_ovrtx_preview_fallbacks_to_stage_file("missing.usd") == 0


def test_looks_scope_helpers_cover_empty_relative_and_missing_specs() -> None:
    stage = Usd.Stage.CreateInMemory()

    material_utils.ensure_looks_scope(stage, "")
    material_utils.ensure_looks_scope(stage, "relative/Looks/Mat")
    material_utils._author_looks_scope_type(stage, Sdf.Path("/Root/Looks"))

    looks_spec = stage.GetEditTarget().GetLayer().GetPrimAtPath("/Root/Looks")
    assert looks_spec.typeName == "Scope"


def test_asset_resolution_helpers_cover_file_urls_and_missing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_file = tmp_path / "surface.mdl"
    local_file.write_text("// mdl", encoding="utf-8")

    assert material_utils._resolve_local_asset_path(
        object(),
        local_file.as_uri(),
        tmp_path,
    ) == (str(local_file.resolve()), True)
    assert material_utils._resolve_local_asset_path(
        object(),
        "file://server/share/surface.mdl",
        tmp_path,
    ) == (None, False)
    assert material_utils._resolve_local_asset_path(
        object(),
        "file:///C:/missing/surface.mdl",
        tmp_path,
    ) == (None, False)
    assert material_utils._resolve_local_asset_path(
        object(),
        str(local_file),
        tmp_path,
    ) == (str(local_file.resolve()), True)
    assert material_utils._resolve_local_asset_path(
        object(),
        str(tmp_path / "missing.mdl"),
        tmp_path,
    ) == (None, False)
    assert material_utils._resolve_local_asset_path(
        object(),
        "surface.mdl",
        tmp_path,
    ) == (str(local_file.resolve()), True)
    assert material_utils._resolve_local_asset_path(
        object(),
        "missing.mdl",
        tmp_path,
    ) == (None, False)
    with monkeypatch.context() as patch:
        patch.setattr(
            material_utils.Path,
            "exists",
            lambda self: (_ for _ in ()).throw(OSError("bad path")),
        )
        assert material_utils._safe_exists("bad-path") is False
    assert material_utils._is_non_local_asset_uri("https://example.com/a.mdl")


def test_mdl_asset_discovery_covers_remote_empty_and_stringified_values(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    no_attr = UsdShade.Shader.Define(stage, "/World/NoAttr")
    assert no_attr
    unset = UsdShade.Shader.Define(stage, "/World/Unset")
    unset.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset)
    empty = UsdShade.Shader.Define(stage, "/World/Empty")
    empty.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(""))
    remote = UsdShade.Shader.Define(stage, "/World/Remote")
    remote.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("https://example.com/material.mdl"))

    assert material_utils.get_local_mdl_assets(stage, base_dir=tmp_path) == [
        {
            "shader_path": "/World/Remote",
            "mdl_path": "https://example.com/material.mdl",
            "resolved_path": None,
            "is_local": False,
        },
    ]
    assert material_utils.get_local_mdl_assets(stage) == [
        {
            "shader_path": "/World/Remote",
            "mdl_path": "https://example.com/material.mdl",
            "resolved_path": None,
            "is_local": False,
        },
    ]

    file_stage_path = tmp_path / "scene.usda"
    file_stage = Usd.Stage.CreateNew(str(file_stage_path))
    file_shader = UsdShade.Shader.Define(file_stage, "/FileShader")
    file_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("surface.mdl"))
    (tmp_path / "surface.mdl").write_text("// mdl", encoding="utf-8")
    assert material_utils.get_local_mdl_assets(file_stage) == [
        {
            "shader_path": "/FileShader",
            "mdl_path": "surface.mdl",
            "resolved_path": str((tmp_path / "surface.mdl").resolve()),
            "is_local": True,
        },
    ]

    class _PathRaises:
        @property
        def path(self) -> str:
            raise RuntimeError("path failed")

        def __str__(self) -> str:
            return "fallback.mdl"

    class _FakeMdlAttr:
        def IsValid(self) -> bool:
            return True

        def Get(self) -> _PathRaises:
            return _PathRaises()

    class _FakeShaderPrim:
        def IsA(self, schema: object) -> bool:
            return schema is UsdShade.Shader

        def GetAttribute(self, name: str) -> _FakeMdlAttr:
            return _FakeMdlAttr()

        def GetPath(self) -> str:
            return "/FakeShader"

    class _FakeStage:
        def Traverse(self) -> list[_FakeShaderPrim]:
            return [_FakeShaderPrim()]

    assert material_utils.get_local_mdl_assets(_FakeStage(), base_dir=tmp_path) == [
        {
            "shader_path": "/FakeShader",
            "mdl_path": "fallback.mdl",
            "resolved_path": None,
            "is_local": False,
        },
    ]
    assert set(
        material_utils.get_unique_mdl_directories(
            [
                {"is_local": False, "resolved_path": None},
                {"is_local": True, "resolved_path": str(tmp_path / "a" / "one.mdl")},
                {"is_local": True, "resolved_path": str(tmp_path / "a" / "two.mdl")},
            ],
        ),
    ) == {tmp_path / "a"}


def test_package_texture_localization_covers_skip_and_extract_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert material_utils._resolve_package_asset_path("asset.usdz", tmp_path).name == (
        "asset.usdz"
    )

    class _FakeRootLayer:
        realPath = ""
        identifier = ""

    class _FakeInstanceProxyPrim:
        def IsInstanceProxy(self) -> bool:
            return True

    class _FakeStage:
        def GetRootLayer(self) -> _FakeRootLayer:
            return _FakeRootLayer()

        def Traverse(self) -> list[_FakeInstanceProxyPrim]:
            return [_FakeInstanceProxyPrim()]

    assert (
        material_utils.localize_package_texture_assets_for_render(
            _FakeStage(),
            tmp_path / "out",
        )
        == 0
    )

    skip_stage = Usd.Stage.CreateInMemory()
    skip_material = UsdShade.Material.Define(skip_stage, "/World/Looks/Skip")
    skip_material.GetPrim().CreateAttribute("inputs:unset", Sdf.ValueTypeNames.Asset)
    skip_material.GetPrim().CreateAttribute(
        "inputs:empty",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(""))
    skip_material.GetPrim().CreateAttribute(
        "inputs:not_package",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("plain.png"))
    non_texture_package = tmp_path / "non_texture.usdz"
    with zipfile.ZipFile(non_texture_package, "w") as archive:
        archive.writestr("0/readme.txt", "hello")
    skip_material.GetPrim().CreateAttribute(
        "inputs:not_texture",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(f"{non_texture_package}[0/readme.txt]"))
    assert (
        material_utils.localize_package_texture_assets_for_render(
            skip_stage,
            tmp_path / "skip-out",
        )
        == 0
    )

    package = tmp_path / "asset.usdz"
    source_image = _make_texture(tmp_path / "source.png", (1, 2, 3))
    with zipfile.ZipFile(package, "w") as archive:
        archive.write(source_image, "0/source.png")

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    attr = material.GetPrim().CreateAttribute(
        "inputs:file",
        Sdf.ValueTypeNames.Asset,
    )
    attr.Set(Sdf.AssetPath(f"{package}[0/source.png]"))

    monkeypatch.setattr(
        material_utils,
        "extract_usdz_member_to_path",
        lambda *args, **kwargs: None,
    )
    assert (
        material_utils.localize_package_texture_assets_for_render(
            stage,
            tmp_path / "out-none",
        )
        == 0
    )
    assert attr.Get().path == f"{package}[0/source.png]"

    def _raise_extract(*args: object, **kwargs: object) -> None:
        raise RuntimeError("extract failed")

    monkeypatch.setattr(material_utils, "extract_usdz_member_to_path", _raise_extract)
    assert (
        material_utils.localize_package_texture_assets_for_render(
            stage,
            tmp_path / "out-error",
        )
        == 0
    )


def test_texture_asset_discovery_covers_duplicates_and_stringified_values(
    tmp_path: Path,
) -> None:
    texture = _make_texture(tmp_path / "texture.png")
    stage = Usd.Stage.CreateInMemory()
    first = UsdShade.Shader.Define(stage, "/World/First")
    first.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    duplicate = UsdShade.Shader.Define(stage, "/World/Duplicate")
    duplicate.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture)),
    )
    remote = UsdShade.Shader.Define(stage, "/World/Remote")
    remote.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("http://example.com/texture.png"),
    )
    text = UsdShade.Shader.Define(stage, "/World/Text")
    text.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("notes.txt"))
    empty = UsdShade.Shader.Define(stage, "/World/Empty")
    empty.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(""))

    assets = material_utils.get_local_texture_file_assets(stage)

    assert [asset["prim_path"] for asset in assets] == ["/World/First", "/World/Remote"]
    assert assets[0]["resolved_path"] == str(texture.resolve())
    assert assets[1]["is_local"] is False

    file_stage_path = tmp_path / "texture_scene.usda"
    file_stage = Usd.Stage.CreateNew(str(file_stage_path))
    relative_shader = UsdShade.Shader.Define(file_stage, "/Relative")
    relative_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("texture.png"),
    )
    assert material_utils.get_local_texture_file_assets(file_stage) == [
        {
            "prim_path": "/Relative",
            "attr_name": "inputs:file",
            "file_path": "texture.png",
            "resolved_path": str(texture.resolve()),
            "is_local": True,
        },
    ]

    class _TypeName:
        class type:
            typeName = "SdfAssetPath"

    class _PathRaises:
        @property
        def path(self) -> str:
            raise RuntimeError("path failed")

        def __str__(self) -> str:
            return "fallback.png"

    class _FakeTextureAttr:
        def __init__(self, value: object) -> None:
            self._value = value

        def GetTypeName(self) -> _TypeName:
            return _TypeName()

        def Get(self) -> object:
            return self._value

        def GetName(self) -> str:
            return "inputs:file"

    class _FakeTexturePrim:
        def __init__(self, value: object) -> None:
            self._value = value

        def GetAttributes(self) -> list[_FakeTextureAttr]:
            return [_FakeTextureAttr(self._value)]

        def GetPath(self) -> str:
            return "/FakeTexture"

    class _FakeTextureStage:
        def Traverse(self) -> list[_FakeTexturePrim]:
            return [_FakeTexturePrim(None), _FakeTexturePrim(_PathRaises())]

        def GetRootLayer(self) -> object:
            return types.SimpleNamespace(realPath="")

    assert material_utils.get_local_texture_file_assets(
        _FakeTextureStage(),
        base_dir=tmp_path,
    ) == [
        {
            "prim_path": "/FakeTexture",
            "attr_name": "inputs:file",
            "file_path": "fallback.png",
            "resolved_path": None,
            "is_local": False,
        },
    ]


def test_add_mdl_material_and_bind_material_cover_path_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    _, material_path = material_utils.add_mdl_material(
        stage,
        "Paint",
        "OmniPBR.mdl",
        color="#ffffff",
    )
    assert material_path == "/World/Looks/Paint"

    fallback_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(fallback_stage, "/Root")
    _, fallback_path = material_utils.add_mdl_material(
        fallback_stage,
        "Paint",
        "OmniPBR.mdl",
    )
    assert fallback_path == "/Root/Looks/Paint"

    empty_stage = Usd.Stage.CreateInMemory()
    _, root_path = material_utils.add_mdl_material(
        empty_stage,
        "Paint",
        "OmniPBR.mdl",
    )
    assert root_path == "/Looks/Paint"

    assert (
        material_utils.bind_material_to_prim(stage, material_path, "/Missing") is stage
    )
    mesh = _define_triangle_mesh(stage, "/World/Mesh")
    material_utils.bind_material_to_prim(
        stage,
        material_path,
        "/World/Mesh",
        binding_strength=UsdShade.Tokens.strongerThanDescendants,
    )
    bound = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0]
    assert str(bound.GetPath()) == material_path

    usdex_pkg = types.ModuleType("usdex")
    usdex_core = types.ModuleType("usdex.core")
    calls: list[tuple[Usd.Prim, UsdShade.Material]] = []

    def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> bool:
        calls.append((prim, material))
        return True

    usdex_core.bindMaterial = _bind_material
    usdex_pkg.core = usdex_core
    monkeypatch.setitem(sys.modules, "usdex", usdex_pkg)
    monkeypatch.setitem(sys.modules, "usdex.core", usdex_core)
    material_utils.bind_material_to_prim(stage, material_path, "/World/Mesh")
    assert calls

    class _FailingBindingAPI:
        @staticmethod
        def Apply(prim: Usd.Prim) -> object:
            raise RuntimeError("bind failed")

    monkeypatch.setattr(
        material_utils.UsdShade, "MaterialBindingAPI", _FailingBindingAPI
    )
    material_utils.bind_material_to_prim(
        stage,
        material_path,
        "/World/Mesh",
        binding_strength=UsdShade.Tokens.strongerThanDescendants,
    )

    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    UsdGeom.Xform.Define(prototype_stage, "/Prototype")
    _define_triangle_mesh(prototype_stage, "/Prototype/Geom")
    prototype_stage.GetRootLayer().Save()

    instance_stage = Usd.Stage.CreateInMemory()
    material_utils.add_mdl_material(instance_stage, "Paint", "OmniPBR.mdl")
    inst = UsdGeom.Xform.Define(instance_stage, "/Instance")
    inst.GetPrim().GetReferences().AddReference(str(prototype_path), "/Prototype")
    inst.GetPrim().SetInstanceable(True)
    assert instance_stage.GetPrimAtPath("/Instance/Geom").IsInstanceProxy()
    with pytest.raises(ValueError, match="instance proxy"):
        material_utils.bind_material_to_prim(
            instance_stage,
            "/Looks/Paint",
            "/Instance/Geom",
        )


def test_convert_custom_mdl_to_builtin_covers_omnipbr_and_triplanar_variants() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdShade.Shader.Define(stage, "/World/NoAttr")
    blank = UsdShade.Shader.Define(stage, "/World/Blank")
    blank.GetPrim().CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset)
    local_omni = UsdShade.Shader.Define(stage, "/World/LocalOmni")
    local_omni.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("./Material/OmniPBR.mdl"))
    other = UsdShade.Shader.Define(stage, "/World/Other")
    other.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("Other.mdl"))

    triplanar = UsdShade.Shader.Define(stage, "/World/Tri")
    triplanar.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("CreativePBRTriplanar.mdl"))
    triplanar.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier",
        Sdf.ValueTypeNames.Token,
    ).Set("CreativePBRTriplanar")
    triplanar.CreateInput("diffuse_color_a", Sdf.ValueTypeNames.Color3f).Set(
        (0.1, 0.2, 0.3),
    )
    triplanar.CreateInput("roughness_b", Sdf.ValueTypeNames.Float)
    triplanar.CreateInput("already_plain", Sdf.ValueTypeNames.Float).Set(0.7)

    material_utils.convert_custom_mdl_to_builtin(stage)

    assert (
        local_omni.GetPrim().GetAttribute("info:mdl:sourceAsset").Get().path
        == "OmniPBR.mdl"
    )
    assert (
        other.GetPrim().GetAttribute("info:mdl:sourceAsset").Get().path == "Other.mdl"
    )
    assert (
        triplanar.GetPrim().GetAttribute("info:mdl:sourceAsset").Get().path
        == "OmniPBR.mdl"
    )
    assert (
        triplanar.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
        == "OmniPBR"
    )
    assert triplanar.GetInput("diffuse_color").Get() == pytest.approx(
        (0.1, 0.2, 0.3),
    )
    assert not triplanar.GetInput("roughness")
    assert triplanar.GetInput("already_plain").Get() == pytest.approx(0.7)
