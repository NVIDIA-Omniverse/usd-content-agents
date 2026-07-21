# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material discovery functions."""

from pathlib import Path

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from texture_agent.functions import material_discovery as material_discovery_module
from texture_agent.functions.material_discovery import (
    EffectiveMaterialDiscovery,
    MaterialInfo,
    PrimTextureUnit,
    _add_material_alias,
    _add_material_binding,
    _find_bound_prims,
    _material_library_owner_path,
    _should_fold_subset_material_into_parent,
    discover_effective_materials,
    discover_materials,
    discover_materials_from_file,
    expand_to_prim_units,
)


def _create_stage_with_material(
    base_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    metalness: float = 1.0,
    roughness: float = 0.3,
    texture_file: str | None = None,
    material_name: str = "TestMaterial",
) -> Usd.Stage:
    """Create an in-memory USD stage with a sphere + OpenPBR material."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Create world
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    # Create geometry
    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    sphere.GetRadiusAttr().Set(1.0)

    # Create Looks scope and material
    UsdGeom.Scope.Define(stage, "/World/Looks")
    mat_path = f"/World/Looks/{material_name}"
    material = UsdShade.Material.Define(stage, mat_path)

    # Set OpenPBR inputs on the material prim
    mat_prim = material.GetPrim()
    mat_prim.CreateAttribute("inputs:base_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*base_color)
    )
    mat_prim.CreateAttribute("inputs:base_metalness", Sdf.ValueTypeNames.Float).Set(
        metalness
    )
    mat_prim.CreateAttribute("inputs:specular_roughness", Sdf.ValueTypeNames.Float).Set(
        roughness
    )

    tex_path = texture_file if texture_file else ""
    mat_prim.CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(tex_path))

    # Bind material to sphere
    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
    binding_api.Bind(material)

    return stage


def _create_stage_with_mdl_material() -> Usd.Stage:
    """Create an in-memory USD stage with a bound SimReady-style MDL material."""
    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    material = UsdShade.Material.Define(stage, "/World/Looks/Plastic")

    shader = UsdShade.Shader.Define(stage, "/World/Looks/Plastic/Shader")
    shader_prim = shader.GetPrim()
    shader_prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://simready.example/Plastic.mdl")
    )
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.1, 0.2, 0.3)
    )
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://simready.example/T_Plastic_Normal.png")
    )
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://simready.example/T_Plastic_ORM.png")
    )

    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
    binding_api.Bind(material)

    return stage


def _create_stage_with_mdl_over_shader_material() -> Usd.Stage:
    """Create a material with shader metadata authored on a typed over."""
    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    material = UsdShade.Material.Define(stage, "/World/Looks/PlasticOver")

    shader_prim = stage.OverridePrim("/World/Looks/PlasticOver/Shader")
    shader_prim.SetTypeName("Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.3, 0.4)
    )
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://simready.example/T_PlasticOver_Normal.png")
    )

    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
    binding_api.Bind(material)

    return stage


def _create_stage_with_materialx_texture_reader(
    input_name: str = "file",
) -> Usd.Stage:
    """Create a material with a MaterialX-style image node file input."""
    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    material = UsdShade.Material.Define(stage, "/World/Looks/MaterialXPlastic")

    shader = UsdShade.Shader.Define(
        stage,
        "/World/Looks/MaterialXPlastic/diffuse_texture",
    )
    shader.CreateIdAttr("ND_image_color3")
    shader.CreateInput(input_name, Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://materialx.example/T_Plastic_BaseColor.png")
    )

    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
    binding_api.Bind(material)

    return stage


def _create_stage_with_invalid_shader_float() -> Usd.Stage:
    """Create a material with one invalid and one valid shader roughness input."""
    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    material = UsdShade.Material.Define(stage, "/World/Looks/StringFloat")

    bad_shader = UsdShade.Shader.Define(stage, "/World/Looks/StringFloat/BadShader")
    bad_shader.CreateInput("roughness", Sdf.ValueTypeNames.String).Set("rough")
    bad_shader.CreateInput("metalness", Sdf.ValueTypeNames.String).Set("metal")

    good_shader = UsdShade.Shader.Define(stage, "/World/Looks/StringFloat/GoodShader")
    good_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)

    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim())
    binding_api.Bind(material)

    return stage


def _create_stage_with_many_bound_materials(count: int) -> Usd.Stage:
    """Create an in-memory USD stage with one bound material per mesh."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.Scope.Define(stage, "/World/Looks")

    for i in range(count):
        cube = UsdGeom.Cube.Define(stage, f"/World/Mesh_{i:04d}")
        cube.CreateSizeAttr(0.01)
        mat = UsdShade.Material.Define(stage, f"/World/Looks/Mat_{i:04d}")
        shader = UsdShade.Shader.Define(
            stage, f"/World/Looks/Mat_{i:04d}/PreviewSurface"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.1, 0.3, 0.1)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    return stage


class TestDiscoverMaterials:
    """Tests for discover_materials()."""

    def test_discovers_single_material(self) -> None:
        """Discovers a single material with correct properties."""
        stage = _create_stage_with_material(
            base_color=(0.9, 0.6, 0.5),
            metalness=1.0,
            roughness=0.15,
            material_name="Copper",
        )

        materials = discover_materials(stage)

        assert len(materials) == 1
        mat = materials[0]
        assert mat.name == "Copper"
        assert mat.prim_path == "/World/Looks/Copper"
        assert pytest.approx(mat.base_color[0], abs=0.01) == 0.9
        assert pytest.approx(mat.base_color[1], abs=0.01) == 0.6
        assert pytest.approx(mat.base_color[2], abs=0.01) == 0.5
        assert mat.base_metalness == pytest.approx(1.0)
        assert mat.specular_roughness == pytest.approx(0.15)
        assert mat.has_existing_texture is False
        assert len(mat.bound_prim_paths) == 1
        assert mat.bound_prim_paths[0] == "/World/Sphere"

    def test_add_material_binding_ignores_invalid_material_handle(self) -> None:
        bindings = {}

        _add_material_binding(bindings, UsdShade.Material())
        _add_material_binding(
            bindings,
            type(
                "TruthyInvalidMaterial",
                (),
                {"__bool__": lambda self: True, "GetPrim": lambda self: None},
            )(),
        )

        assert bindings == {}

    def test_subset_alias_helper_defensive_edges(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        parent_material = UsdShade.Material.Define(stage, "/World/Looks/Semantic")
        subset_material = UsdShade.Material.Define(
            stage,
            "/World/Assembly/Part/Looks/Diffuse",
        )

        bindings = {}
        _add_material_alias(bindings, UsdShade.Material(), subset_material)
        assert bindings == {}

        invalid_material = type(
            "TruthyInvalidMaterial",
            (),
            {"__bool__": lambda self: True, "GetPrim": lambda self: None},
        )()
        _add_material_alias(bindings, parent_material, invalid_material)
        assert bindings == {}

        assert _material_library_owner_path("/World/NoScope/Mat") is None
        assert (
            _should_fold_subset_material_into_parent(
                subset_material=UsdShade.Material(),
                parent_material=parent_material,
                subset_path="/World/Assembly/Part/Mesh/face_0",
                default_prim_path=None,
            )
            is False
        )
        assert (
            _should_fold_subset_material_into_parent(
                subset_material=invalid_material,
                parent_material=parent_material,
                subset_path="/World/Assembly/Part/Mesh/face_0",
                default_prim_path=None,
            )
            is False
        )
        assert (
            _should_fold_subset_material_into_parent(
                subset_material=subset_material,
                parent_material=subset_material,
                subset_path="/World/Assembly/Part/Mesh/face_0",
                default_prim_path=None,
            )
            is False
        )

        no_scope_material = UsdShade.Material.Define(
            stage,
            "/World/Assembly/Part/Diffuse",
        )
        assert (
            _should_fold_subset_material_into_parent(
                subset_material=no_scope_material,
                parent_material=parent_material,
                subset_path="/World/Assembly/Part/Mesh/face_0",
                default_prim_path=None,
            )
            is False
        )

        top_level_subset = UsdShade.Material.Define(stage, "/World/Looks/Diffuse")
        assert (
            _should_fold_subset_material_into_parent(
                subset_material=top_level_subset,
                parent_material=parent_material,
                subset_path="/World/Mesh/face_0",
                default_prim_path="/World",
            )
            is False
        )

        default_scope_subset = UsdShade.Material.Define(
            stage,
            "/World/Assembly/Looks/Diffuse",
        )
        assert (
            _should_fold_subset_material_into_parent(
                subset_material=default_scope_subset,
                parent_material=parent_material,
                subset_path="/World/Assembly/Mesh/face_0",
                default_prim_path="/World/Assembly",
            )
            is False
        )

        local_parent = UsdShade.Material.Define(
            stage,
            "/World/Assembly/Part/Looks/Semantic",
        )
        assert (
            _should_fold_subset_material_into_parent(
                subset_material=subset_material,
                parent_material=local_parent,
                subset_path="/World/Assembly/Part/Mesh/face_0",
                default_prim_path="/World",
            )
            is False
        )

    def test_detects_existing_texture(self) -> None:
        """Correctly flags materials that already have a texture file."""
        stage = _create_stage_with_material(
            texture_file="/path/to/albedo.png",
            material_name="Textured",
        )

        materials = discover_materials(stage)

        assert len(materials) == 1
        assert materials[0].has_existing_texture is True
        assert materials[0].base_color_texture == "/path/to/albedo.png"

    def test_empty_texture_is_not_existing(self) -> None:
        """Empty texture path ('') is treated as no texture."""
        stage = _create_stage_with_material(
            texture_file="",
            material_name="NoTexture",
        )

        materials = discover_materials(stage)

        assert len(materials) == 1
        assert materials[0].has_existing_texture is False
        assert materials[0].base_color_texture is None

    def test_discovers_mdl_shader_properties(self) -> None:
        """Reads SimReady/MDL shader inputs when OpenPBR attrs are absent."""
        stage = _create_stage_with_mdl_material()

        materials = discover_materials(stage)

        assert len(materials) == 1
        mat = materials[0]
        assert mat.name == "Plastic"
        assert mat.base_color == pytest.approx((0.1, 0.2, 0.3))
        assert mat.has_existing_texture is True
        assert mat.base_color_texture is None
        assert mat.bound_prim_paths == ["/World/Sphere"]

    def test_discovers_typed_over_shader_properties(self) -> None:
        """Reads shader inputs authored on typed over descendants."""
        stage = _create_stage_with_mdl_over_shader_material()

        materials = discover_materials(stage)

        assert len(materials) == 1
        mat = materials[0]
        assert mat.name == "PlasticOver"
        assert mat.base_color == pytest.approx((0.2, 0.3, 0.4))
        assert mat.has_existing_texture is True
        assert mat.base_color_texture is None

    def test_discovers_materialx_file_texture_reader(self) -> None:
        """Reads albedo texture paths from MaterialX image node file inputs."""
        stage = _create_stage_with_materialx_texture_reader()

        materials = discover_materials(stage)

        assert len(materials) == 1
        mat = materials[0]
        assert mat.name == "MaterialXPlastic"
        assert mat.has_existing_texture is True
        assert (
            mat.base_color_texture
            == "omniverse://materialx.example/T_Plastic_BaseColor.png"
        )

    def test_discovers_materialx_filename_texture_reader(self) -> None:
        """Reads albedo texture paths from MaterialX filename inputs."""
        stage = _create_stage_with_materialx_texture_reader(input_name="filename")

        materials = discover_materials(stage)

        assert len(materials) == 1
        mat = materials[0]
        assert mat.has_existing_texture is True
        assert (
            mat.base_color_texture
            == "omniverse://materialx.example/T_Plastic_BaseColor.png"
        )

    def test_ignores_invalid_shader_float_inputs(self) -> None:
        """Invalid shader float-like inputs do not abort material discovery."""
        stage = _create_stage_with_invalid_shader_float()

        materials = discover_materials(stage)

        assert len(materials) == 1
        mat = materials[0]
        assert mat.name == "StringFloat"
        assert mat.base_metalness is None
        assert mat.specular_roughness == pytest.approx(0.42)

    def test_multiple_materials(self) -> None:
        """Discovers multiple materials in one stage."""
        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        UsdGeom.Scope.Define(stage, "/World/Looks")

        for name, color in [("Steel", (0.3, 0.3, 0.3)), ("Gold", (1.0, 0.8, 0.3))]:
            mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            mat.GetPrim().CreateAttribute(
                "inputs:base_color", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(*color))
            mat.GetPrim().CreateAttribute(
                "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
            ).Set(Sdf.AssetPath(""))

        materials = discover_materials(stage)

        assert len(materials) == 2
        names = {m.name for m in materials}
        assert names == {"Steel", "Gold"}

    def test_ladder_fixture_reports_shader_backed_materials(self) -> None:
        """Regression coverage for the shipped ladder asset."""
        fixture = (
            Path(__file__).resolve().parents[1]
            / "data/examples/ladder/sources/usd/ladder.usd"
        )

        materials = {m.name: m for m in discover_materials_from_file(fixture)}

        assert set(materials) == {
            "Aluminum_Brushed",
            "Aluminum_Matte",
            "Plastic_Dark_Blue",
            "Rubber_Black_Matte",
        }

        rubber = materials["Rubber_Black_Matte"]
        assert rubber.has_existing_texture is False
        assert rubber.bound_prim_paths == [
            "/RootNode/Geometry/M_AluminumStepLadder_B01_Rubber"
        ]

        plastic_dark_blue = materials["Plastic_Dark_Blue"]
        assert plastic_dark_blue.base_color != pytest.approx((0.5, 0.5, 0.5))
        assert plastic_dark_blue.bound_prim_paths == [
            "/RootNode/Geometry/M_AluminumStepLadder_B01_Plastic2"
        ]

    def test_prim_path_filter(self) -> None:
        """prim_paths filter restricts which materials are returned."""
        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        UsdGeom.Scope.Define(stage, "/World/Looks")
        for name in ["MatA", "MatB", "MatC"]:
            mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            mat.GetPrim().CreateAttribute(
                "inputs:base_color", Sdf.ValueTypeNames.Color3f
            ).Set(Gf.Vec3f(0.5, 0.5, 0.5))

        materials = discover_materials(stage, prim_paths=["/World/Looks/MatB"])

        assert len(materials) == 1
        assert materials[0].name == "MatB"

    def test_no_materials(self) -> None:
        """Returns empty list when no materials exist."""
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/World", "Xform")

        materials = discover_materials(stage)

        assert materials == []

    def test_large_material_discovery_uses_one_pass_binding_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Large CAD-like stages should not traverse the whole stage per material."""
        stage = _create_stage_with_many_bound_materials(600)
        calls = 0

        original = material_discovery_module._build_material_bound_prim_index

        def counted_binding_index(stage_arg: Usd.Stage) -> dict[str, list[str]]:
            nonlocal calls
            calls += 1
            return original(stage_arg)

        monkeypatch.setattr(
            material_discovery_module,
            "_build_material_bound_prim_index",
            counted_binding_index,
        )
        materials = discover_materials(stage)

        assert len(materials) == 600
        assert materials[0].bound_prim_paths == ["/World/Mesh_0000"]
        assert materials[-1].bound_prim_paths == ["/World/Mesh_0599"]
        assert calls == 1

    def test_find_bound_prims_wrapper_uses_binding_index(self) -> None:
        """Compatibility wrapper should return indexed bound prim paths."""
        stage = _create_stage_with_many_bound_materials(2)

        assert _find_bound_prims(stage, "/World/Looks/Mat_0001") == ["/World/Mesh_0001"]
        assert _find_bound_prims(stage, "/World/Looks/Missing") == []


class TestDiscoverEffectiveMaterials:
    """Tests for deterministic effective-bound material discovery."""

    def test_instance_proxies_reduce_to_shared_prototype_material(
        self, tmp_path: Path
    ) -> None:
        model_path = tmp_path / "instanced_model.usda"
        model_stage = Usd.Stage.CreateNew(str(model_path))
        model = UsdGeom.Xform.Define(model_stage, "/Model")
        model_stage.SetDefaultPrim(model.GetPrim())
        cube = UsdGeom.Cube.Define(model_stage, "/Model/Cube")
        material = UsdShade.Material.Define(
            model_stage,
            "/Model/Looks/Shared",
        )
        UsdShade.Material.Define(model_stage, "/Model/Looks/Unused")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        model_stage.GetRootLayer().Save()

        stage = Usd.Stage.CreateInMemory()
        for instance_path in ("/World/InstanceB", "/World/InstanceA"):
            instance = UsdGeom.Xform.Define(stage, instance_path).GetPrim()
            instance.GetReferences().AddReference(str(model_path), "/Model")
            instance.SetInstanceable(True)

        discovery = discover_effective_materials(stage)

        assert isinstance(discovery, EffectiveMaterialDiscovery)
        assert discovery.authored_material_count == 2
        assert discovery.renderable_prim_count == 2
        assert discovery.effective_bound_material_count == 1
        material_info = discovery.effective_materials[0]
        assert material_info.prim_path == "/World/InstanceA/Looks/Shared"
        assert material_info.material_alias_paths == [
            "/World/InstanceA/Looks/Shared",
            "/World/InstanceB/Looks/Shared",
        ]
        assert material_info.bound_prim_paths == [
            "/World/InstanceA/Cube",
            "/World/InstanceB/Cube",
        ]
        assert len(discovery.skipped_materials) == 1
        assert discovery.skipped_materials[0].material_prim_path == (
            "/World/InstanceA/Looks/Unused"
        )
        assert discovery.skipped_materials[0].reason_code == "not_effectively_bound"

    def test_subsets_duplicate_names_and_unbound_materials_are_auditable(
        self,
    ) -> None:
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        shared_a = UsdShade.Material.Define(stage, "/World/Looks/A/Shared")
        shared_b = UsdShade.Material.Define(stage, "/World/Looks/B/Shared")
        UsdShade.Material.Define(stage, "/World/Looks/Unused")

        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        binding_api.Bind(shared_a)
        subset = binding_api.CreateMaterialBindSubset("painted", [0])
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(shared_b)

        discovery = discover_effective_materials(stage)

        assert [m.prim_path for m in discovery.authored_materials] == [
            "/World/Looks/A/Shared",
            "/World/Looks/B/Shared",
            "/World/Looks/Unused",
        ]
        assert [m.name for m in discovery.effective_materials] == [
            "Shared",
            "Shared",
        ]
        assert discovery.renderable_prim_paths == ("/World/Mesh",)
        assert discovery.renderable_subset_paths == ("/World/Mesh/painted",)
        assert discovery.effective_materials[0].bound_prim_paths == ["/World/Mesh"]
        assert discovery.effective_materials[1].bound_subset_paths == [
            "/World/Mesh/painted"
        ]
        assert [skip.reason_code for skip in discovery.skipped_materials] == [
            "not_effectively_bound"
        ]
        assert discovery.skipped_materials[0].material_prim_path == (
            "/World/Looks/Unused"
        )

    def test_component_local_subset_materials_fold_into_upstream_semantic_material(
        self,
    ) -> None:
        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)
        mesh = UsdGeom.Mesh.Define(stage, "/World/Assembly/Part/Mesh")
        semantic = UsdShade.Material.Define(stage, "/World/Looks/Copper")
        subset_clone = UsdShade.Material.Define(
            stage,
            "/World/Assembly/Part/Looks/Diffuse_44",
        )

        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        binding_api.Bind(semantic)
        subset = binding_api.CreateMaterialBindSubset("face_0", [0])
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(subset_clone)

        discovery = discover_effective_materials(stage)

        assert discovery.authored_material_count == 2
        assert discovery.effective_bound_material_count == 1
        material = discovery.effective_materials[0]
        assert material.prim_path == "/World/Looks/Copper"
        assert material.bound_prim_paths == ["/World/Assembly/Part/Mesh"]
        assert material.bound_subset_paths == ["/World/Assembly/Part/Mesh/face_0"]
        assert material.material_alias_paths == [
            "/World/Assembly/Part/Looks/Diffuse_44",
            "/World/Looks/Copper",
        ]
        assert [skip.material_prim_path for skip in discovery.skipped_materials] == [
            "/World/Assembly/Part/Looks/Diffuse_44",
        ]

    def test_explicit_and_upstream_scopes_filter_effective_membership(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        material_a = UsdShade.Material.Define(stage, "/World/Looks/A")
        material_b = UsdShade.Material.Define(stage, "/World/Looks/B")
        mesh_a = UsdGeom.Cube.Define(stage, "/World/Assembly/A/Mesh")
        mesh_b = UsdGeom.Cube.Define(stage, "/World/Assembly/B/Mesh")
        UsdShade.MaterialBindingAPI.Apply(mesh_a.GetPrim()).Bind(material_a)
        UsdShade.MaterialBindingAPI.Apply(mesh_b.GetPrim()).Bind(material_b)

        explicit = discover_effective_materials(
            stage,
            material_prim_paths=["/World/Looks/B"],
            prim_scope_paths=["/World/Assembly/B"],
        )
        assert [m.prim_path for m in explicit.effective_materials] == ["/World/Looks/B"]
        assert explicit.renderable_prim_paths == ("/World/Assembly/B/Mesh",)
        assert explicit.skipped_materials[0].reason_code == "outside_material_scope"

        upstream = discover_effective_materials(
            stage,
            upstream_assignment_paths=["/World/Assembly/A"],
        )
        assert [m.prim_path for m in upstream.effective_materials] == ["/World/Looks/A"]
        assert upstream.effective_materials[0].bound_prim_paths == [
            "/World/Assembly/A/Mesh"
        ]
        assert upstream.skipped_materials[0].reason_code == (
            "outside_upstream_assignment_scope"
        )

    def test_explicit_prim_scope_reports_outside_prim_scope_skip(self) -> None:
        stage = _create_stage_with_material(material_name="ScopedPaint")

        discovery = discover_effective_materials(
            stage,
            prim_scope_paths=["/World/OtherMesh"],
        )

        assert discovery.effective_materials == ()
        assert len(discovery.skipped_materials) == 1
        assert discovery.skipped_materials[0].reason_code == "outside_prim_scope"

    def test_siemens_expected_counts_need_no_backend(self) -> None:
        """Mirror the Siemens count contract with cheap in-memory prims."""
        stage = Usd.Stage.CreateInMemory()
        materials = [
            UsdShade.Material.Define(stage, f"/World/Looks/Mat_{index:04d}")
            for index in range(5592)
        ]
        for index in range(4283):
            mesh = stage.DefinePrim(f"/World/Geometry/Mesh_{index:04d}", "Mesh")
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(materials[index % 10])

        discovery = discover_effective_materials(stage)

        assert discovery.authored_material_count == 5592
        assert discovery.renderable_prim_count == 4283
        assert discovery.effective_bound_material_count == 10
        assert len(discovery.skipped_materials) == 5582
        assert {skip.reason_code for skip in discovery.skipped_materials} == {
            "not_effectively_bound"
        }


class TestExpandToPrimUnits:
    """Tests for expand_to_prim_units()."""

    def _make_material(self, name: str, bound: list[str] | None = None) -> MaterialInfo:
        return MaterialInfo(
            prim_path=f"/World/Looks/{name}",
            name=name,
            bound_prim_paths=bound or [],
            base_color=(0.5, 0.5, 0.5),
        )

    def test_per_material_mode(self) -> None:
        """Per-material mode creates one unit per material."""
        materials = [
            self._make_material("Steel", ["/World/A", "/World/B"]),
            self._make_material("Copper", ["/World/C"]),
        ]
        specs = {
            "Steel": {"prompt": "rusty steel", "opacity": 0.8},
            "Copper": {"prompt": "patina copper", "opacity": 0.7},
        }

        units = expand_to_prim_units(materials, specs, mode="per_material")

        assert len(units) == 2
        assert units[0].key == "Steel"
        assert units[0].prim_path == ""
        assert units[1].key == "Copper"
        assert units[0].detail_policy == "default"
        assert units[0].prompt == "rusty steel"

    def test_surface_only_policy_adds_prompt_guard(self) -> None:
        """surface_only appends guardrails against semantic modeled detail."""
        materials = [
            self._make_material("Plastic_Green", ["/World/PCB"]),
        ]
        specs = {
            "Plastic_Green": {
                "prompt": (
                    "green PCB solder mask material with printed copper traces, "
                    "vias, component pads, silkscreen labels and realistic "
                    "circuit board markings"
                ),
                "opacity": 0.55,
            }
        }

        units = expand_to_prim_units(
            materials,
            specs,
            mode="per_material",
            default_detail_policy="surface_only",
        )

        assert len(units) == 1
        assert units[0].detail_policy == "surface_only"
        assert "Surface-only material texture:" in units[0].prompt
        assert "green solder mask material" in units[0].prompt
        assert "Avoid traces, vias, pads" in units[0].prompt
        description = units[0].prompt.split(". Avoid", maxsplit=1)[0]
        for forbidden in (
            "PCB",
            "copper",
            "traces",
            "vias",
            "pads",
            "silkscreen",
            "labels",
            "circuit",
            "board",
            "markings",
        ):
            assert forbidden.lower() not in description.lower()

    def test_prompt_none_is_empty_before_detail_policy(self) -> None:
        materials = [self._make_material("Plastic_Green", ["/World/PCB"])]
        specs = {
            "Plastic_Green": {
                "prompt": None,
                "detail_policy": "surface_only",
            }
        }

        units = expand_to_prim_units(materials, specs, mode="per_material")

        assert units[0].detail_policy == "surface_only"
        assert "plain continuous material surface" in units[0].prompt

    def test_prompt_type_errors_name_config_key(self) -> None:
        materials = [self._make_material("Steel", ["/World/Rail_L"])]

        with pytest.raises(
            ValueError,
            match=r"material_textures\.Steel\.prompt must be a string",
        ):
            expand_to_prim_units(
                materials,
                {"Steel": {"prompt": 123}},
                mode="per_material",
            )

        with pytest.raises(
            ValueError,
            match=(
                r"material_textures\.Steel\.per_prim\./World/Rail_L"
                r"\.prompt must be a string"
            ),
        ):
            expand_to_prim_units(
                materials,
                {
                    "Steel": {
                        "prompt": "rusty steel",
                        "per_prim": {
                            "/World/Rail_L": {"prompt": ["bad"]},
                        },
                    }
                },
                mode="per_prim",
            )

    def test_per_prim_mode(self) -> None:
        """Per-prim mode creates one unit per bound prim."""
        materials = [
            self._make_material("Steel", ["/World/Rail_L", "/World/Rail_R"]),
        ]
        specs = {"Steel": {"prompt": "rusty steel", "opacity": 0.8}}

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 2
        assert units[0].key == "Steel__Rail_L"
        assert units[0].prim_path == "/World/Rail_L"
        assert units[1].key == "Steel__Rail_R"
        assert units[1].prim_path == "/World/Rail_R"
        # Different seeds
        assert units[0].seed != units[1].seed

    def test_per_prim_with_overrides(self) -> None:
        """Per-prim overrides provide per-prim prompts."""
        materials = [
            self._make_material("Steel", ["/World/Rail_L", "/World/Rail_R"]),
        ]
        specs = {
            "Steel": {
                "prompt": "rusty steel",
                "opacity": 0.8,
                "per_prim": {
                    "/World/Rail_L": {
                        "prompt": "heavily rusted left rail",
                        "opacity": 0.95,
                    }
                },
            }
        }

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 2
        left = next(u for u in units if "Rail_L" in u.key)
        right = next(u for u in units if "Rail_R" in u.key)
        assert left.prompt == "heavily rusted left rail"
        assert left.opacity == 0.95
        assert right.prompt == "rusty steel"  # inherits from parent
        assert right.opacity == 0.8

    def test_per_prim_detail_policy_overrides_material_policy(self) -> None:
        """Per-prim policy can inherit or override the material policy."""
        materials = [
            self._make_material("Plastic_Green", ["/World/PCB_A", "/World/PCB_B"]),
        ]
        specs = {
            "Plastic_Green": {
                "prompt": "matte green solder mask",
                "opacity": 0.6,
                "detail_policy": "surface_only",
                "per_prim": {
                    "/World/PCB_B": {"detail_policy": "default"},
                },
            }
        }

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        by_prim = {unit.prim_path: unit for unit in units}
        assert by_prim["/World/PCB_A"].detail_policy == "surface_only"
        assert "Avoid traces, vias, pads" in by_prim["/World/PCB_A"].prompt
        assert by_prim["/World/PCB_B"].detail_policy == "default"
        assert by_prim["/World/PCB_B"].prompt == "matte green solder mask"

    def test_skips_materials_without_spec(self) -> None:
        """Materials not in material_textures are skipped."""
        materials = [
            self._make_material("Steel", ["/World/A"]),
            self._make_material("Unknown", ["/World/B"]),
        ]
        specs = {"Steel": {"prompt": "rusty", "opacity": 0.8}}

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 1
        assert units[0].key == "Steel__A"

    def test_no_bound_prims_per_prim(self) -> None:
        """Material with no bound prims in per-prim mode falls back to per-material."""
        materials = [self._make_material("Steel", [])]
        specs = {"Steel": {"prompt": "rusty", "opacity": 0.8}}

        units = expand_to_prim_units(materials, specs, mode="per_prim")

        assert len(units) == 1
        assert units[0].key == "Steel"
        assert units[0].prim_path == ""
