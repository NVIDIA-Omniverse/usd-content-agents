# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for created-material OpenPBR authoring and evidence."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdShade  # noqa: E402

import material_agent.material_library_generation.creation as creation_module  # noqa: E402
import material_agent.material_library_generation.usd_authoring as usd_authoring_module  # noqa: E402
from material_agent.material_library_generation.creation import (  # noqa: E402
    MaterialCreationBackendRegistry,
    create_material_package,
)
from material_agent.material_library_generation.creation_contract import (  # noqa: E402
    CreateMaterialRequest,
    MaterialCreationError,
    MaterialCreationErrorCode,
)
from material_agent.material_library_generation.fake_backend import (  # noqa: E402
    FakeMaterialCreationBackend,
)
from material_agent.material_library_generation.schema import (  # noqa: E402
    GeneratedMaterial,
    MaterialRecipe,
    PBRHints,
    TextureMapSet,
)
from material_agent.material_library_generation.usd_authoring import (  # noqa: E402
    MaterialAuthoringContractError,
    inspect_material_library_authoring,
    probe_openpbr_materialx_authoring,
    write_material_library_usd,
)


def _recipe() -> MaterialRecipe:
    return MaterialRecipe(
        id="wp9g_brushed_metal",
        name="WP9G Brushed Metal",
        description="Brushed metal used to verify portable OpenPBR authoring.",
        appearance_prompt="fine brushed silver metal",
        color="silver",
        material="aluminum",
        finish="brushed",
        base_color_hint=(0.55, 0.58, 0.62),
        pbr_hints=PBRHints(roughness=0.34, metallic=1.0),
    )


def _generated_material(tmp_path: Path) -> GeneratedMaterial:
    texture_dir = tmp_path / "textures" / _recipe().material_id
    texture_dir.mkdir(parents=True)
    textures = TextureMapSet(
        albedo=texture_dir / "albedo.png",
        normal=texture_dir / "normal.png",
        orm=texture_dir / "orm.png",
    )
    for path in (textures.albedo, textures.normal, textures.orm):
        path.write_bytes(b"texture")
    return GeneratedMaterial(recipe=_recipe(), textures=textures)


def _request(tmp_path: Path) -> CreateMaterialRequest:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    return CreateMaterialRequest(
        source_usd=source,
        target_prim_paths=("/World/Asset",),
        recipe=_recipe(),
        backend="fake",
        source_usd_sha256="0" * 64,
    )


def _registry(backend: FakeMaterialCreationBackend) -> MaterialCreationBackendRegistry:
    registry = MaterialCreationBackendRegistry()
    registry.register(backend, make_default=True)
    return registry


def _assert_contract_error(
    expected_code: str,
    operation: Callable[[], Any],
) -> MaterialAuthoringContractError:
    with pytest.raises(MaterialAuthoringContractError) as raised:
        operation()
    assert raised.value.code == expected_code
    return raised.value


def _install_incomplete_usdex(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_usdex = types.ModuleType("usdex")
    fake_core = types.ModuleType("usdex.core")
    fake_core.definePbrMaterial = lambda *args, **kwargs: None
    fake_usdex.core = fake_core
    monkeypatch.setitem(sys.modules, "usdex", fake_usdex)
    monkeypatch.setitem(sys.modules, "usdex.core", fake_core)


def _install_textured_openpbr_usdex(
    monkeypatch: pytest.MonkeyPatch,
    *,
    albedo_color_space: str = "sRGB",
    albedo_shader_id: str = "ND_tiledimage_color3",
    albedo_output: str = "out",
    definition_returns_none: bool = False,
    include_ao_output: bool = True,
    normalmap_shader_id: str = "ND_normalmap",
    roughness_output: str = "outg",
    surface_output: str = "out",
    texcoord_shader_id: str = "ND_texcoord_vector2",
    texcoord_value: int | str = 0,
) -> None:
    fake_usdex = types.ModuleType("usdex")
    fake_core = types.ModuleType("usdex.core")

    def texcoord(stage: Usd.Stage, material_path: str) -> UsdShade.Shader:
        shader = UsdShade.Shader.Get(stage, f"{material_path}/Texcoord")
        if shader and shader.GetPrim().IsValid():
            return shader
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Texcoord")
        shader.CreateIdAttr(texcoord_shader_id)
        if texcoord_shader_id == "ND_geompropvalue_vector2":
            shader.CreateInput("geomprop", Sdf.ValueTypeNames.String).Set(
                str(texcoord_value)
            )
        else:
            shader.CreateInput("index", Sdf.ValueTypeNames.Int).Set(int(texcoord_value))
        shader.CreateOutput("out", Sdf.ValueTypeNames.Float2)
        return shader

    def image(
        material: UsdShade.Material,
        name: str,
        shader_id: str,
        output_type: Any,
        path: Sdf.AssetPath,
        color_space: str,
    ) -> UsdShade.Shader:
        stage = material.GetPrim().GetStage()
        material_path = str(material.GetPath())
        shader = UsdShade.Shader.Define(stage, f"{material_path}/{name}")
        shader.CreateIdAttr(shader_id)
        file_input = shader.CreateInput("file", Sdf.ValueTypeNames.Asset)
        file_input.Set(path)
        file_input.GetAttr().SetColorSpace(color_space)
        shader.CreateInput("texcoord", Sdf.ValueTypeNames.Float2).ConnectToSource(
            texcoord(stage, material_path).GetOutput("out")
        )
        shader.CreateOutput("out", output_type)
        return shader

    def surface(material: UsdShade.Material) -> UsdShade.Shader:
        stage = material.GetPrim().GetStage()
        return UsdShade.Shader.Get(stage, f"{material.GetPath()}/OpenPBR")

    def define_pbr_material(
        stage: Usd.Stage,
        path: Sdf.Path,
        color: Gf.Vec3f,
        opacity: float,
        roughness: float,
        metallic: float,
    ) -> UsdShade.Material | None:
        if definition_returns_none:
            return None
        material = UsdShade.Material.Define(stage, path)
        material.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(color)
        material.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(
            roughness
        )
        material.CreateInput("geometry_opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        material.CreateInput("transmission_weight", Sdf.ValueTypeNames.Float).Set(0.0)
        openpbr = UsdShade.Shader.Define(stage, path.AppendChild("OpenPBR"))
        openpbr.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
        material.CreateSurfaceOutput("mtlx").ConnectToSource(
            openpbr.CreateOutput(surface_output, Sdf.ValueTypeNames.Token)
        )
        return material

    def add_diffuse(material: UsdShade.Material, path: Sdf.AssetPath) -> None:
        texture = image(
            material,
            "Albedo",
            albedo_shader_id,
            Sdf.ValueTypeNames.Color3f,
            path,
            albedo_color_space,
        )
        if albedo_output != "out":
            texture.CreateOutput(albedo_output, Sdf.ValueTypeNames.Color3f)
        surface(material).CreateInput(
            "base_color", Sdf.ValueTypeNames.Color3f
        ).ConnectToSource(texture.GetOutput(albedo_output))

    def add_normal(material: UsdShade.Material, path: Sdf.AssetPath) -> None:
        texture = image(
            material,
            "Normal",
            "ND_tiledimage_vector3",
            Sdf.ValueTypeNames.Float3,
            path,
            "raw",
        )
        stage = material.GetPrim().GetStage()
        normalmap = UsdShade.Shader.Define(stage, f"{material.GetPath()}/NormalMap")
        normalmap.CreateIdAttr(normalmap_shader_id)
        normalmap.CreateInput("in", Sdf.ValueTypeNames.Float3).ConnectToSource(
            texture.GetOutput("out")
        )
        normalmap.CreateOutput("out", Sdf.ValueTypeNames.Float3)
        surface(material).CreateInput(
            "geometry_normal", Sdf.ValueTypeNames.Float3
        ).ConnectToSource(normalmap.GetOutput("out"))

    def add_orm(material: UsdShade.Material, path: Sdf.AssetPath) -> None:
        texture = image(
            material,
            "ORM",
            "ND_tiledimage_color3",
            Sdf.ValueTypeNames.Color3f,
            path,
            "raw",
        )
        stage = material.GetPrim().GetStage()
        separate = UsdShade.Shader.Define(stage, f"{material.GetPath()}/SeparateORM")
        separate.CreateIdAttr("ND_separate3_color3")
        separate.CreateInput("in", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.GetOutput("out")
        )
        output_names = (
            ("outr", "outg", "outb") if include_ao_output else ("outg", "outb")
        )
        for output_name in output_names:
            separate.CreateOutput(output_name, Sdf.ValueTypeNames.Float)
        surface(material).CreateInput(
            "specular_roughness", Sdf.ValueTypeNames.Float
        ).ConnectToSource(separate.GetOutput(roughness_output))
        surface(material).CreateInput(
            "base_metalness", Sdf.ValueTypeNames.Float
        ).ConnectToSource(separate.GetOutput("outb"))

    fake_core.definePbrMaterial = define_pbr_material
    fake_core.addDiffuseTextureToPbrMaterial = add_diffuse
    fake_core.addNormalTextureToPbrMaterial = add_normal
    fake_core.addOrmTextureToPbrMaterial = add_orm
    fake_usdex.core = fake_core
    monkeypatch.setitem(sys.modules, "usdex", fake_usdex)
    monkeypatch.setitem(sys.modules, "usdex.core", fake_core)


def test_openpbr_capability_requires_definition_and_texture_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        usd_authoring_module.metadata,
        "version",
        lambda _distribution: "2.3.0",
    )
    _install_incomplete_usdex(monkeypatch)

    capability = probe_openpbr_materialx_authoring()

    assert capability.available is False
    assert capability.installed_version == "2.3.0"
    assert capability.missing_symbols == (
        "usdex.core.addDiffuseTextureToPbrMaterial",
        "usdex.core.addNormalTextureToPbrMaterial",
        "usdex.core.addOrmTextureToPbrMaterial",
    )


def test_openpbr_capability_reports_native_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_name: str) -> NoReturn:
        raise RuntimeError("native USD-Exchange load failed")

    monkeypatch.setattr(
        usd_authoring_module.importlib,
        "import_module",
        fail_import,
    )

    capability = probe_openpbr_materialx_authoring()

    assert capability.available is False
    assert capability.import_error == ("RuntimeError: native USD-Exchange load failed")


def test_openpbr_capability_reports_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_distribution: str) -> str:
        raise usd_authoring_module.metadata.PackageNotFoundError

    monkeypatch.setattr(
        usd_authoring_module.metadata,
        "version",
        missing_distribution,
    )
    _install_incomplete_usdex(monkeypatch)

    capability = probe_openpbr_materialx_authoring()

    assert capability.available is False
    assert capability.installed_version is None


def test_connection_guards_reject_missing_ambiguous_and_non_shader_sources() -> None:
    class SourcesPort:
        def __init__(
            self,
            sources: list[object],
            invalid_sources: list[object] | None = None,
        ) -> None:
            self.sources = sources
            self.invalid_sources = invalid_sources or []

        def GetConnectedSources(self) -> tuple[list[object], list[object]]:
            return self.sources, self.invalid_sources

    _assert_contract_error(
        "MATERIAL_CONNECTION_MISSING",
        lambda: usd_authoring_module._connected_source(
            None,
            label="test.input",
            material_path="/World/Looks/Test",
        ),
    )
    ambiguous = _assert_contract_error(
        "MATERIAL_CONNECTION_INVALID",
        lambda: usd_authoring_module._connected_source(
            SourcesPort([object(), object()]),
            label="test.input",
            material_path="/World/Looks/Test",
        ),
    )
    assert ambiguous.details["source_count"] == 2
    invalid = _assert_contract_error(
        "MATERIAL_CONNECTION_INVALID",
        lambda: usd_authoring_module._connected_source(
            SourcesPort([object()], [Sdf.Path("/Missing.outputs:out")]),
            label="test.input",
            material_path="/World/Looks/Test",
        ),
    )
    assert invalid.details["invalid_source_paths"] == ["/Missing.outputs:out"]

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/Looks/Test")
    source = material.CreateInput("source", Sdf.ValueTypeNames.Float)
    target = material.CreateInput("target", Sdf.ValueTypeNames.Float)
    target.ConnectToSource(source)
    _assert_contract_error(
        "MATERIAL_SHADER_SOURCE_INVALID",
        lambda: usd_authoring_module._connected_shader(
            target,
            label="test.target",
            material_path=str(material.GetPath()),
        ),
    )


def test_material_interface_input_resolution_guards() -> None:
    class FakeAttr:
        def __init__(self, path: str, color_space: str = "") -> None:
            self.path = path
            self.color_space = color_space

        def GetPath(self) -> str:
            return self.path

        def GetColorSpace(self) -> str:
            return self.color_space

    class FakeInput:
        def __init__(
            self,
            path: str,
            *,
            value: Any = None,
            color_space: str = "",
        ) -> None:
            self.attr = FakeAttr(path, color_space)
            self.value = value
            self.sources: list[object] = []
            self.invalid_sources: list[object] = []

        def GetAttr(self) -> FakeAttr:
            return self.attr

        def GetConnectedSources(self) -> tuple[list[object], list[object]]:
            return self.sources, self.invalid_sources

        def Get(self) -> Any:
            return self.value

    class FakeConnectable:
        def __init__(self, target: FakeInput | None) -> None:
            self.target = target

        def GetInput(self, _name: str) -> FakeInput | None:
            return self.target

    def source_info(
        target: FakeInput | None,
        *,
        source_type: Any = UsdShade.AttributeType.Input,
    ) -> object:
        return types.SimpleNamespace(
            sourceType=source_type,
            source=FakeConnectable(target),
            sourceName="value",
        )

    resolved = FakeInput(
        "/World/Looks/Test.inputs:resolved",
        value=Sdf.AssetPath("textures/albedo.png"),
        color_space="sRGB",
    )
    interface = FakeInput("/World/Looks/Test.inputs:interface")
    interface.sources = [source_info(resolved)]
    value, color_space = usd_authoring_module._resolved_input_value(
        interface,
        label="albedo.file",
        material_path="/World/Looks/Test",
    )
    assert value == Sdf.AssetPath("textures/albedo.png")
    assert color_space == "sRGB"

    _assert_contract_error(
        "MATERIAL_INPUT_MISSING",
        lambda: usd_authoring_module._resolved_input_value(
            None,
            label="test.missing_input",
            material_path="/World/Looks/Test",
        ),
    )

    invalid_authored = FakeInput("/World/Looks/Test.inputs:invalid_authored")
    invalid_authored.invalid_sources = [Sdf.Path("/Missing.inputs:value")]
    invalid_error = _assert_contract_error(
        "MATERIAL_INPUT_CONNECTION_INVALID",
        lambda: usd_authoring_module._resolved_input_value(
            invalid_authored,
            label="test.invalid_authored",
            material_path="/World/Looks/Test",
        ),
    )
    assert invalid_error.details["invalid_source_paths"] == ["/Missing.inputs:value"]

    invalid_type = FakeInput("/World/Looks/Test.inputs:invalid_type")
    invalid_type.sources = [
        source_info(resolved, source_type=UsdShade.AttributeType.Output)
    ]
    _assert_contract_error(
        "MATERIAL_INPUT_CONNECTION_INVALID",
        lambda: usd_authoring_module._resolved_input_value(
            invalid_type,
            label="test.invalid_type",
            material_path="/World/Looks/Test",
        ),
    )

    missing = FakeInput("/World/Looks/Test.inputs:missing")
    missing.sources = [source_info(None)]
    _assert_contract_error(
        "MATERIAL_INPUT_CONNECTION_INVALID",
        lambda: usd_authoring_module._resolved_input_value(
            missing,
            label="test.missing",
            material_path="/World/Looks/Test",
        ),
    )

    cycle = FakeInput("/World/Looks/Test.inputs:cycle")
    cycle.sources = [source_info(cycle)]
    _assert_contract_error(
        "MATERIAL_INPUT_CONNECTION_CYCLE",
        lambda: usd_authoring_module._resolved_input_value(
            cycle,
            label="test.cycle",
            material_path="/World/Looks/Test",
        ),
    )

    chain = [FakeInput(f"/World/Looks/Test.inputs:depth_{index}") for index in range(9)]
    for current, target in zip(chain[:-1], chain[1:], strict=True):
        current.sources = [source_info(target)]
    _assert_contract_error(
        "MATERIAL_INPUT_CONNECTION_DEPTH_EXCEEDED",
        lambda: usd_authoring_module._resolved_input_value(
            chain[0],
            label="test.depth",
            material_path="/World/Looks/Test",
        ),
    )


def test_invalid_material_profile_fails_before_backend(
    tmp_path: Path,
) -> None:
    backend = FakeMaterialCreationBackend()
    package_dir = tmp_path / "package"

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(
            _request(tmp_path),
            package_dir,
            registry=_registry(backend),
            material_profile="not_a_material_profile",
        )

    assert raised.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert backend.calls == []
    assert not package_dir.exists()


def test_display_color_profile_fails_before_backend(tmp_path: Path) -> None:
    backend = FakeMaterialCreationBackend()
    package_dir = tmp_path / "package"

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(
            _request(tmp_path),
            package_dir,
            registry=_registry(backend),
            material_profile="display_color",
        )

    assert raised.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert "apply-only" in str(raised.value)
    assert backend.calls == []
    assert not package_dir.exists()


def test_explicit_openpbr_fails_before_backend_when_usdex_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_incomplete_usdex(monkeypatch)
    backend = FakeMaterialCreationBackend()
    package_dir = tmp_path / "package"

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(
            _request(tmp_path),
            package_dir,
            registry=_registry(backend),
            material_profile="openpbr_materialx",
        )

    assert raised.value.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert raised.value.retryable is False
    assert raised.value.diagnostics[0].code == (
        "OPENPBR_MATERIALX_AUTHORING_UNAVAILABLE"
    )
    assert raised.value.diagnostics[0].details["dependency_issue"] == 371
    assert backend.calls == []
    assert not package_dir.exists()


def test_openpbr_preflight_does_not_delete_existing_package_on_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_incomplete_usdex(monkeypatch)
    backend = FakeMaterialCreationBackend()
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    marker = package_dir / "existing.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(
            _request(tmp_path),
            package_dir,
            registry=_registry(backend),
            material_profile="openpbr_materialx",
            overwrite=True,
        )

    assert raised.value.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert backend.calls == []


def test_cached_openpbr_package_reuses_without_authoring_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_textured_openpbr_usdex(monkeypatch)
    backend = FakeMaterialCreationBackend()
    registry = _registry(backend)
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    create_material_package(
        request,
        package_dir,
        registry=registry,
        material_profile="openpbr_materialx",
    )
    _install_incomplete_usdex(monkeypatch)

    cached = create_material_package(
        request,
        package_dir,
        registry=registry,
        material_profile="openpbr_materialx",
    )

    assert cached.validation["cache_hit"] is True
    assert cached.validation["material_profile"]["resolved_profile"] == (
        "openpbr_materialx"
    )
    assert backend.calls == [request.request_id]


def test_cached_package_wraps_unexpected_authoring_inspection_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = FakeMaterialCreationBackend()
    registry = _registry(backend)
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    create_material_package(request, package_dir, registry=registry)

    def fail_inspection(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError("USD inspection failed unexpectedly")

    monkeypatch.setattr(
        creation_module,
        "inspect_material_library_authoring",
        fail_inspection,
    )

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(request, package_dir, registry=registry)

    assert raised.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert raised.value.diagnostics[0].code == "CACHED_AUTHORING_INSPECTION_FAILED"
    assert raised.value.diagnostics[0].phase == "cache_validation"
    assert backend.calls == [request.request_id]


def test_authoring_contract_failure_removes_partial_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = FakeMaterialCreationBackend()
    package_dir = tmp_path / "package"

    def fail_authoring(*_args: Any, **_kwargs: Any) -> None:
        raise MaterialAuthoringContractError(
            "TEST_AUTHORING_CONTRACT_FAILURE",
            "synthetic authoring contract failure",
            details={"reason": "test"},
        )

    monkeypatch.setattr(
        creation_module,
        "write_material_library_usd",
        fail_authoring,
    )

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(
            _request(tmp_path),
            package_dir,
            registry=_registry(backend),
        )

    assert raised.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert raised.value.diagnostics[0].code == "TEST_AUTHORING_CONTRACT_FAILURE"
    assert raised.value.diagnostics[0].phase == "authoring_validation"
    assert not package_dir.exists()


def test_textured_openpbr_graph_records_authoritative_profile_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_textured_openpbr_usdex(monkeypatch)
    generated = _generated_material(tmp_path)
    library_path = tmp_path / "material_library.usda"

    write_material_library_usd(
        library_path,
        (generated,),
        material_profile="openpbr_materialx",
    )
    evidence = inspect_material_library_authoring(
        library_path,
        (generated,),
        material_profile="openpbr_materialx",
    )

    assert evidence["requested_profile"] == "openpbr_materialx"
    assert evidence["resolved_profile"] == "openpbr_materialx"
    assert evidence["stage_reopened"] is True
    material = evidence["materials"][generated.binding]
    assert material["authoritative_output"] == "outputs:mtlx:surface"
    assert material["authoritative_shader_id"] == ("ND_open_pbr_surface_surfaceshader")
    assert material["compatibility_outputs"] == ["outputs:surface"]
    assert material["textures"]["albedo"]["color_space"] == "sRGB"
    assert material["textures"]["albedo"]["uv_primvar"] == "st"
    assert material["textures"]["orm"]["color_space"] == "raw"
    assert material["textures"]["orm"]["connections"] == {
        "r": "ambient_occlusion_provenance_only",
        "g": "specular_roughness",
        "b": "base_metalness",
    }
    assert material["textures"]["normal"]["normal_convention"] == (
        "tangent_opengl_positive_y"
    )
    assert material["textures"]["normal"]["normalmap_shader_id"] == ("ND_normalmap")

    stage = Usd.Stage.Open(str(library_path))
    assert stage is not None
    openpbr = UsdShade.Material(stage.GetPrimAtPath(generated.binding))
    assert openpbr.GetSurfaceOutput("mtlx").HasConnectedSource()
    preview = UsdShade.Shader(
        stage.GetPrimAtPath(f"{generated.binding}/OVRTXPreviewSurface")
    )
    assert preview.GetIdAttr().Get() == "UsdPreviewSurface"


def test_explicit_openpbr_authors_generated_maps_instead_of_prototype(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_textured_openpbr_usdex(monkeypatch)
    generated = _generated_material(tmp_path)
    prototype_path = tmp_path / "prototype.usda"
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    prototype_material = UsdShade.Material.Define(
        prototype_stage,
        "/World/Looks/Prototype",
    )
    prototype_shader = UsdShade.Shader.Define(
        prototype_stage,
        "/World/Looks/Prototype/PrototypePreview",
    )
    prototype_shader.CreateIdAttr("UsdPreviewSurface")
    prototype_material.CreateSurfaceOutput().ConnectToSource(
        prototype_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    prototype_stage.GetRootLayer().Save()
    generated = GeneratedMaterial(
        recipe=generated.recipe,
        textures=generated.textures,
        prototype_source={
            "library_path": prototype_path,
            "binding": "/World/Looks/Prototype",
        },
    )
    library_path = tmp_path / "material_library.usda"

    write_material_library_usd(
        library_path,
        (generated,),
        material_profile="openpbr_materialx",
    )

    evidence = inspect_material_library_authoring(
        library_path,
        (generated,),
        material_profile="openpbr_materialx",
    )
    assert evidence["resolved_profile"] == "openpbr_materialx"
    stage = Usd.Stage.Open(str(library_path))
    assert stage is not None
    assert not stage.GetPrimAtPath(f"{generated.binding}/PrototypePreview").IsValid()


def test_explicit_openpbr_rejects_invalid_material_from_definition_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_textured_openpbr_usdex(
        monkeypatch,
        definition_returns_none=True,
    )

    _assert_contract_error(
        "OPENPBR_MATERIALX_DEFINITION_FAILED",
        lambda: write_material_library_usd(
            tmp_path / "material_library.usda",
            (_generated_material(tmp_path),),
            material_profile="openpbr_materialx",
        ),
    )


def test_empty_texture_reference_is_rejected(tmp_path: Path) -> None:
    error = _assert_contract_error(
        "MATERIAL_TEXTURE_REFERENCE_MISSING",
        lambda: usd_authoring_module._validate_portable_texture_path(
            "",
            library_path=tmp_path / "material_library.usda",
            material_path="/World/Looks/Test",
            channel="albedo",
        ),
    )
    assert error.details["channel"] == "albedo"


def test_materialx_image_rejects_non_generated_texture_path(tmp_path: Path) -> None:
    stage = Usd.Stage.CreateInMemory()
    image = UsdShade.Shader.Define(stage, "/Image")
    image.CreateIdAttr("ND_tiledimage_color3")
    file_input = image.CreateInput("file", Sdf.ValueTypeNames.Asset)
    file_input.Set(Sdf.AssetPath("textures/not-generated.png"))
    file_input.GetAttr().SetColorSpace("sRGB")

    _assert_contract_error(
        "OPENPBR_TEXTURE_REFERENCE_MISMATCH",
        lambda: usd_authoring_module._validate_materialx_image(
            image,
            expected_path="textures/generated.png",
            expected_color_space="sRGB",
            library_path=tmp_path / "material_library.usda",
            material_path="/World/Looks/Test",
            channel="albedo",
        ),
    )


def test_materialx_geomprop_st_is_a_valid_uv_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_textured_openpbr_usdex(
        monkeypatch,
        texcoord_shader_id="ND_geompropvalue_vector2",
        texcoord_value="st",
    )

    write_material_library_usd(
        tmp_path / "material_library.usda",
        (_generated_material(tmp_path),),
        material_profile="openpbr_materialx",
    )


@pytest.mark.parametrize(
    ("authoring_options", "error_code"),
    [
        ({"albedo_shader_id": "ND_constant_color3"}, "OPENPBR_TEXTURE_NODE_INVALID"),
        (
            {"texcoord_shader_id": "ND_texcoord_vector2", "texcoord_value": 1},
            "OPENPBR_UV_BINDING_INVALID",
        ),
        (
            {
                "texcoord_shader_id": "ND_geompropvalue_vector2",
                "texcoord_value": "uv0",
            },
            "OPENPBR_UV_BINDING_INVALID",
        ),
        (
            {"texcoord_shader_id": "ND_unknown_vector2", "texcoord_value": 0},
            "OPENPBR_UV_BINDING_INVALID",
        ),
        ({"surface_output": "wrong"}, "OPENPBR_SURFACE_SHADER_INVALID"),
        ({"include_ao_output": False}, "OPENPBR_ORM_CONNECTION_INVALID"),
        (
            {"normalmap_shader_id": "ND_passthrough_vector3"},
            "OPENPBR_NORMAL_CONNECTION_INVALID",
        ),
    ],
)
def test_textured_openpbr_graph_rejects_invalid_nodes_paths_and_uvs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    authoring_options: dict[str, object],
    error_code: str,
) -> None:
    _install_textured_openpbr_usdex(monkeypatch, **authoring_options)

    _assert_contract_error(
        error_code,
        lambda: write_material_library_usd(
            tmp_path / "material_library.usda",
            (_generated_material(tmp_path),),
            material_profile="openpbr_materialx",
        ),
    )


def test_openpbr_graph_requires_mtlx_surface_and_ovrtx_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from world_understanding.utils.usd import material as usd_material_module

    generated = _generated_material(tmp_path)
    empty_stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(empty_stage, generated.binding)
    _assert_contract_error(
        "OPENPBR_SURFACE_MISSING",
        lambda: usd_authoring_module._validate_openpbr_materialx_graph(
            material,
            generated=generated,
            library_path=tmp_path / "material_library.usda",
        ),
    )

    _install_textured_openpbr_usdex(monkeypatch)
    monkeypatch.setattr(
        usd_material_module,
        "add_ovrtx_preview_fallbacks_for_materialx_openpbr",
        lambda _stage: None,
    )
    _assert_contract_error(
        "OPENPBR_OVRTX_PREVIEW_FALLBACK_MISSING",
        lambda: write_material_library_usd(
            tmp_path / "without_fallback.usda",
            (generated,),
            material_profile="openpbr_materialx",
        ),
    )


@pytest.mark.parametrize(
    ("albedo_color_space", "albedo_output", "roughness_output", "error_code"),
    [
        ("raw", "out", "outg", "OPENPBR_TEXTURE_COLOR_SPACE_INVALID"),
        ("sRGB", "out", "outr", "OPENPBR_ORM_CONNECTION_INVALID"),
        ("sRGB", "wrong", "outg", "MATERIAL_SOURCE_OUTPUT_INVALID"),
    ],
)
def test_textured_openpbr_graph_fails_closed_on_invalid_channel_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    albedo_color_space: str,
    albedo_output: str,
    roughness_output: str,
    error_code: str,
) -> None:
    _install_textured_openpbr_usdex(
        monkeypatch,
        albedo_color_space=albedo_color_space,
        albedo_output=albedo_output,
        roughness_output=roughness_output,
    )

    with pytest.raises(MaterialAuthoringContractError) as raised:
        write_material_library_usd(
            tmp_path / "material_library.usda",
            (_generated_material(tmp_path),),
            material_profile="openpbr_materialx",
        )

    assert raised.value.code == error_code


def test_material_package_records_auto_resolution_without_changing_behavior(
    tmp_path: Path,
) -> None:
    backend = FakeMaterialCreationBackend()

    created = create_material_package(
        _request(tmp_path),
        tmp_path / "package",
        registry=_registry(backend),
    )

    profile = created.validation["material_profile"]
    assert profile["requested_profile"] == "auto"
    assert profile["resolved_profile"] == "preview_surface"
    assert profile["stage_reopened"] is True
    assert profile["texture_reference_count"] == 3
    manifest = json.loads(created.creation_manifest_path.read_text(encoding="utf-8"))
    assert manifest["created_material"]["validation"]["material_profile"] == profile


def test_material_authoring_inspects_omnipbr_profile(tmp_path: Path) -> None:
    generated = _generated_material(tmp_path)
    library_path = tmp_path / "omnipbr.usda"
    stage = Usd.Stage.CreateNew(str(library_path))
    material = UsdShade.Material.Define(stage, generated.binding)
    shader = UsdShade.Shader.Define(stage, f"{generated.binding}/OmniPBR")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    preview = UsdShade.Shader.Define(stage, f"{generated.binding}/PreviewSurface")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    for name, path in (
        ("DiffuseTexture", generated.textures.albedo),
        ("NormalTexture", generated.textures.normal),
        ("ORMTexture", generated.textures.orm),
    ):
        material.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(path.relative_to(library_path.parent).as_posix())
        )
    material.CreateInput("Documentation", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("material-notes.txt")
    )
    stage.GetRootLayer().Save()

    evidence = inspect_material_library_authoring(
        library_path,
        (generated,),
        material_profile="omnipbr_mdl",
    )

    assert evidence["resolved_profile"] == "omnipbr_mdl"
    assert evidence["texture_reference_count"] == 3
    _assert_contract_error(
        "MATERIAL_PROFILE_RESOLUTION_MISMATCH",
        lambda: inspect_material_library_authoring(
            library_path,
            (generated,),
            material_profile="preview_surface",
        ),
    )

    shader.SetSourceAssetSubIdentifier("NotOmniPBR", "mdl")
    stage.GetRootLayer().Save()
    _assert_contract_error(
        "MATERIAL_PROFILE_RESOLUTION_MISMATCH",
        lambda: inspect_material_library_authoring(
            library_path,
            (generated,),
            material_profile="omnipbr_mdl",
        ),
    )


def test_cached_package_rejects_a_different_requested_profile(tmp_path: Path) -> None:
    backend = FakeMaterialCreationBackend()
    registry = _registry(backend)
    package_dir = tmp_path / "package"
    request = _request(tmp_path)
    create_material_package(request, package_dir, registry=registry)

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(
            request,
            package_dir,
            registry=registry,
            material_profile="preview_surface",
        )

    assert raised.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "different material profile" in str(raised.value)
    assert backend.calls == [request.request_id]


def test_cached_package_without_profile_evidence_rejects_explicit_profile(
    tmp_path: Path,
) -> None:
    backend = FakeMaterialCreationBackend()
    registry = _registry(backend)
    package_dir = tmp_path / "package"
    request = _request(tmp_path)
    created = create_material_package(request, package_dir, registry=registry)
    manifest = json.loads(created.creation_manifest_path.read_text(encoding="utf-8"))
    manifest["created_material"]["validation"].pop("material_profile")
    created.creation_manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(MaterialCreationError) as raised:
        create_material_package(
            request,
            package_dir,
            registry=registry,
            material_profile="preview_surface",
        )

    assert raised.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "no material-profile evidence" in str(raised.value)
    assert backend.calls == [request.request_id]


def test_material_inspection_rejects_missing_prim_surface_and_textures(
    tmp_path: Path,
) -> None:
    generated = _generated_material(tmp_path)

    missing_prim_path = tmp_path / "missing-prim.usda"
    missing_prim_stage = Usd.Stage.CreateNew(str(missing_prim_path))
    missing_prim_stage.GetRootLayer().Save()
    _assert_contract_error(
        "MATERIAL_PRIM_MISSING",
        lambda: inspect_material_library_authoring(
            missing_prim_path,
            (generated,),
            material_profile="auto",
        ),
    )

    missing_surface_path = tmp_path / "missing-surface.usda"
    missing_surface_stage = Usd.Stage.CreateNew(str(missing_surface_path))
    UsdShade.Material.Define(missing_surface_stage, generated.binding)
    missing_surface_stage.GetRootLayer().Save()
    _assert_contract_error(
        "MATERIAL_AUTHORITATIVE_SURFACE_MISSING",
        lambda: inspect_material_library_authoring(
            missing_surface_path,
            (generated,),
            material_profile="auto",
        ),
    )

    missing_textures_path = tmp_path / "missing-textures.usda"
    missing_textures_stage = Usd.Stage.CreateNew(str(missing_textures_path))
    material = UsdShade.Material.Define(missing_textures_stage, generated.binding)
    preview = UsdShade.Shader.Define(
        missing_textures_stage,
        f"{generated.binding}/PreviewSurface",
    )
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    missing_textures_stage.GetRootLayer().Save()
    _assert_contract_error(
        "MATERIAL_TEXTURE_REFERENCES_INCOMPLETE",
        lambda: inspect_material_library_authoring(
            missing_textures_path,
            (generated,),
            material_profile="auto",
        ),
    )


def test_material_inspection_reports_stage_reopen_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated = _generated_material(tmp_path)
    monkeypatch.setattr(Usd.Stage, "Open", lambda *_args, **_kwargs: None)

    error = _assert_contract_error(
        "MATERIAL_LIBRARY_STAGE_REOPEN_FAILED",
        lambda: inspect_material_library_authoring(
            tmp_path / "material_library.usda",
            (generated,),
            material_profile="auto",
        ),
    )
    assert error.details["requested_profile"] == "auto"


def test_auto_keeps_input_only_prototype_validation_compatible(tmp_path: Path) -> None:
    generated = _generated_material(tmp_path)
    generated = GeneratedMaterial(
        recipe=generated.recipe,
        textures=generated.textures,
        prototype_source={
            "library_path": "prototype.usda",
            "binding": "/World/Looks/Prototype",
        },
    )
    library_path = tmp_path / "prototype-derived.usda"
    stage = Usd.Stage.CreateNew(str(library_path))
    material = UsdShade.Material.Define(stage, generated.binding)
    material.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.5, 0.5, 0.5)
    )
    stage.GetRootLayer().Save()

    evidence = inspect_material_library_authoring(
        library_path,
        (generated,),
        material_profile="auto",
    )

    assert evidence["requested_profile"] == "auto"
    assert evidence["resolved_profile"] == "prototype_unresolved"
    assert evidence["materials"][generated.binding]["authoritative_output"] is None


def test_auto_keeps_non_openpbr_materialx_prototype_compatible(tmp_path: Path) -> None:
    generated = _generated_material(tmp_path)
    generated = GeneratedMaterial(
        recipe=generated.recipe,
        textures=generated.textures,
        prototype_source={
            "library_path": "prototype.usda",
            "binding": "/World/Looks/Prototype",
        },
    )
    library_path = tmp_path / "prototype-standard-surface.usda"
    stage = Usd.Stage.CreateNew(str(library_path))
    material = UsdShade.Material.Define(stage, generated.binding)
    shader = UsdShade.Shader.Define(stage, f"{generated.binding}/StandardSurface")
    shader.CreateIdAttr("ND_standard_surface_surfaceshader")
    material.CreateSurfaceOutput("mtlx").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()

    evidence = inspect_material_library_authoring(
        library_path,
        (generated,),
        material_profile="auto",
    )

    assert evidence["resolved_profile"] == "prototype_materialx"
    assert evidence["materials"][generated.binding]["authoritative_output"] == (
        "outputs:mtlx:surface"
    )
    _assert_contract_error(
        "OPENPBR_SURFACE_SHADER_INVALID",
        lambda: inspect_material_library_authoring(
            library_path,
            (generated,),
            material_profile="openpbr_materialx",
        ),
    )


def test_auto_keeps_custom_universal_prototype_compatible(tmp_path: Path) -> None:
    generated = _generated_material(tmp_path)
    generated = GeneratedMaterial(
        recipe=generated.recipe,
        textures=generated.textures,
        prototype_source={
            "library_path": "prototype.usda",
            "binding": "/World/Looks/Prototype",
        },
    )
    library_path = tmp_path / "prototype-universal.usda"
    stage = Usd.Stage.CreateNew(str(library_path))
    material = UsdShade.Material.Define(stage, generated.binding)
    shader = UsdShade.Shader.Define(stage, f"{generated.binding}/CustomSurface")
    shader.CreateIdAttr("CustomPrototypeSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()

    evidence = inspect_material_library_authoring(
        library_path,
        (generated,),
        material_profile="auto",
    )

    assert evidence["resolved_profile"] == "prototype_universal"
    assert evidence["materials"][generated.binding]["authoritative_output"] == (
        "outputs:surface"
    )


def test_material_authoring_validation_rejects_removed_texture(
    tmp_path: Path,
) -> None:
    generated = _generated_material(tmp_path)
    library_path = tmp_path / "material_library.usda"
    write_material_library_usd(library_path, (generated,))
    generated.textures.normal.unlink()

    with pytest.raises(MaterialAuthoringContractError) as raised:
        inspect_material_library_authoring(
            library_path,
            (generated,),
            material_profile="auto",
        )

    assert raised.value.code == "MATERIAL_TEXTURE_REFERENCE_UNRESOLVED"


def test_material_authoring_validation_rejects_absolute_texture_reference(
    tmp_path: Path,
) -> None:
    generated = _generated_material(tmp_path)
    library_path = tmp_path / "material_library.usda"
    write_material_library_usd(library_path, (generated,))
    stage = Usd.Stage.Open(str(library_path))
    assert stage is not None
    albedo = UsdShade.Shader(stage.GetPrimAtPath(f"{generated.binding}/AlbedoTexture"))
    albedo.GetInput("file").Set(Sdf.AssetPath(str(generated.textures.albedo)))
    stage.GetRootLayer().Save()

    with pytest.raises(MaterialAuthoringContractError) as raised:
        inspect_material_library_authoring(
            library_path,
            (generated,),
            material_profile="auto",
        )

    assert raised.value.code == "MATERIAL_TEXTURE_REFERENCE_NOT_PORTABLE"
