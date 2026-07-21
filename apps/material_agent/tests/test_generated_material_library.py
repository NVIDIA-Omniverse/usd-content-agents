# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for generated material library package helpers."""

# ruff: noqa: E402,I001

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageStat

pxr = pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

import material_agent.material_library_generation.builder as generated_library_builder  # noqa: E402
import material_agent.material_library_generation.usd_authoring as usd_authoring_module  # noqa: E402
from material_agent.material_library_generation import (  # noqa: E402
    GeneratedMaterial,
    GeneratedMaterialLibrary,
    IntendedPart,
    MaterialGenerationPlan,
    MaterialPrototype,
    MaterialRecipe,
    PBRHints,
    TextureGenerationSettings,
    TextureMapSet,
    build_generated_material_library,
    generate_texture_maps,
    load_material_prototypes_from_data,
    load_material_prototypes_from_manifest,
    make_material_id,
    make_usd_identifier,
    score_material_prototype,
    select_material_prototype,
    validate_generated_material_library,
)
from material_agent.material_library_generation.manifests import (  # noqa: E402
    material_entry,
    write_generation_plan,
    write_materials_manifest,
)
from material_agent.material_library_generation.prototypes import (  # noqa: E402
    _as_color,
    _color_similarity,
    _read_material_input,
)
from material_agent.material_library_generation.texture_generation import (  # noqa: E402
    _create_image_generation_model,
    _deterministic_span_value,
    _match_albedo_mean_to_base_color,
)
from material_agent.tasks.generate_material_library import (  # noqa: E402
    GenerateMaterialLibraryTask,
)
from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask  # noqa: E402
from material_agent.tasks.config_apply import ApplyConfigTask  # noqa: E402
from material_agent.tasks.generate_material_library_config import (  # noqa: E402
    GenerateMaterialLibraryConfigTask,
)
from material_agent.tasks.material_retrieval import MaterialRetrievalTask  # noqa: E402
from material_agent.tasks.resolve_materials import ResolveMaterialFilesTask  # noqa: E402
from material_agent.material_library_generation.validation import _is_nonblank_png  # noqa: E402
from material_agent.materials import (  # noqa: E402
    FALLBACK_MATERIAL_BINDING,
    FALLBACK_MATERIAL_NAME,
)


def _blue_plastic_recipe() -> MaterialRecipe:
    return MaterialRecipe(
        id="blue_glossy_plastic",
        name="Generated Blue Glossy Plastic",
        color="saturated blue",
        material="plastic",
        finish="glossy molded finish",
        description=(
            "Reference-matched saturated blue glossy plastic with subtle molded "
            "texture."
        ),
        appearance_prompt=(
            "saturated blue glossy molded plastic, subtle injection-mold flow "
            "variation, faint scuffs, clean product finish"
        ),
        base_color_hint=(0.02, 0.15, 0.85),
        pbr_hints=PBRHints(metallic=0.0, roughness=0.32),
        intended_parts=(
            IntendedPart(
                semantic_label="outer shell",
                evidence="The reference image shows a saturated blue housing.",
            ),
        ),
    )


def _brushed_aluminum_recipe() -> MaterialRecipe:
    return MaterialRecipe(
        id="brushed_aluminum_fine_grain",
        name="Generated Brushed Aluminum Fine Grain",
        color="silver gray",
        material="aluminum",
        finish="fine brushed satin finish",
        description="Reference-matched brushed aluminum with fine directional grain.",
        appearance_prompt=(
            "silver gray brushed aluminum, fine directional grain, satin "
            "reflections, subtle machining lines"
        ),
        base_color_hint=(0.72, 0.72, 0.70),
        pbr_hints=PBRHints(metallic=1.0, roughness=0.38),
        intended_parts=(IntendedPart(semantic_label="metal rails"),),
    )


def _off_white_satin_plastic_recipe() -> MaterialRecipe:
    return MaterialRecipe(
        id="off_white_satin_plastic",
        name="Off-White Satin Plastic",
        color="off-white",
        material="plastic",
        finish="satin smooth appliance finish",
        description=(
            "Smooth, satin off-white plastic used for the main exterior housing "
            "and lid of laboratory equipment."
        ),
        appearance_prompt=(
            "seamless texture of smooth off-white plastic, satin finish, clean, "
            "minimal wear, medical equipment grade housing"
        ),
        base_color_hint=(0.93, 0.92, 0.90),
        pbr_hints=PBRHints(metallic=0.0, roughness=0.4),
        intended_parts=(IntendedPart(semantic_label="main housing"),),
    )


def _matte_black_rubber_recipe() -> MaterialRecipe:
    return MaterialRecipe(
        id="matte_black_rubber_controls",
        name="Matte Black Rubber Controls",
        color="charcoal black",
        material="rubber",
        finish="matte grip finish",
        description="Dark matte rubber used for rotary control knobs and caps.",
        appearance_prompt=(
            "seamless texture of matte black rubber, non-reflective, subtle "
            "grip texture, solid dark charcoal albedo"
        ),
        base_color_hint=(0.08, 0.08, 0.08),
        pbr_hints=PBRHints(metallic=0.0, roughness=0.7),
        intended_parts=(
            IntendedPart(
                semantic_label="control knobs",
                evidence="The reference controls are black rubber knobs.",
            ),
        ),
    )


def _frosted_glass_recipe() -> MaterialRecipe:
    return MaterialRecipe(
        id="frosted_lid_window",
        name="Frosted Lid Window",
        color="translucent white grey",
        material="frosted glass",
        finish="smooth frosted matte finish",
        description=(
            "Whitish-grey frosted glass or acrylic for a central lid viewing window."
        ),
        appearance_prompt=(
            "seamless frosted translucent glass texture, softly roughened "
            "surface, diffuse light scattering, blurred silhouettes behind"
        ),
        base_color_hint=(0.65, 0.65, 0.65),
        pbr_hints=PBRHints(
            metallic=0.0,
            roughness=0.35,
            transmission=1.0,
            ior=1.48,
        ),
        intended_parts=(IntendedPart(semantic_label="lid window"),),
    )


def _default_materials_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data/materials/material_libs_default/materials.yaml"
    )


def _material_mapping_from_manifest(manifest_path: Path) -> dict[str, str]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    library_path = manifest_path.parent / manifest["library_path"]
    mapping = {"material_library_path": str(library_path)}
    for entry in manifest["entries"]:
        mapping[entry["name"]] = entry["binding"]
    return mapping


def _mean_rgb(path: Path) -> tuple[float, float, float]:
    image = Image.open(path).convert("RGB")
    return tuple(channel / 255.0 for channel in ImageStat.Stat(image).mean)


def _srgb_to_linear_tuple(
    color: tuple[float, float, float],
) -> tuple[float, float, float]:
    def convert(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    return tuple(convert(channel) for channel in color)


def _connected_surface_shader_id(material: UsdShade.Material) -> str | None:
    output = material.GetSurfaceOutput()
    if not output:
        return None
    sources, _ = output.GetConnectedSources()
    if not sources:
        return None
    shader = UsdShade.Shader(sources[0].source.GetPrim())
    return shader.GetIdAttr().Get()


def test_schema_normalizes_ids_and_rejects_duplicates() -> None:
    assert make_material_id(" Blue glossy-plastic! ") == "blue_glossy_plastic"
    assert make_usd_identifier("123 blue glossy-plastic!") == (
        "Material_123_blue_glossy_plastic"
    )

    recipe = _blue_plastic_recipe()
    assert recipe.binding == "/World/Looks/Generated_Blue_Glossy_Plastic"
    unsafe_recipe = MaterialRecipe(
        id="../outside texture",
        name="Unsafe ID",
        description="Material with an unsafe user-provided id.",
        appearance_prompt="neutral material",
    )
    assert unsafe_recipe.material_id == "outside_texture"

    plan = MaterialGenerationPlan(materials=(recipe, recipe))
    with pytest.raises(ValueError, match="duplicate material id"):
        plan.validate()


def test_validation_accepts_opaque_black_png_and_rejects_transparent_png(
    tmp_path: Path,
) -> None:
    black = tmp_path / "black.png"
    transparent = tmp_path / "transparent.png"
    jpeg = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(black)
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(transparent)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(jpeg)

    assert _is_nonblank_png(black) is True
    assert _is_nonblank_png(transparent) is False
    assert _is_nonblank_png(jpeg) is False


def test_validate_generated_library_rejects_non_mapping_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "materials.yaml"
    manifest_path.write_text("- not-a-mapping\n", encoding="utf-8")

    result = validate_generated_material_library(manifest_path)

    assert not result.ok
    assert result.errors == ("materials manifest root must be a mapping",)


def test_apply_config_injects_fallback_into_empty_legacy_mappings() -> None:
    assert ApplyConfigTask._mapping_with_fallback({}) == {
        FALLBACK_MATERIAL_NAME: FALLBACK_MATERIAL_BINDING
    }
    assert ApplyConfigTask._mapping_with_fallback([]) == [
        {FALLBACK_MATERIAL_NAME: FALLBACK_MATERIAL_BINDING}
    ]


def test_generated_material_library_config_rejects_non_mapping_root(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config must be a mapping"):
        GenerateMaterialLibraryConfigTask().run({"config_path": str(config_path)})


def test_generated_material_library_config_resolves_image_lists_from_config_dir(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "generate.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "reference_images": ["../refs/ref.png"],
                "generated_reference_image_paths": "generated/ref.png",
            }
        ),
        encoding="utf-8",
    )

    result = GenerateMaterialLibraryConfigTask().run({"config_path": str(config_path)})

    assert result["reference_images"] == [str((tmp_path / "refs/ref.png").resolve())]
    assert result["generated_reference_image_paths"] == [
        str((config_path.parent / "generated/ref.png").resolve())
    ]


def test_material_planning_prompt_requests_distinct_interior_materials() -> None:
    prompt = GenerateMaterialLibraryTask()._build_planning_prompt(
        {
            "input_usd_path": "/tmp/centrifuge.usd",
            "identification": {
                "asset_type": "laboratory_equipment",
                "asset_subtype": "centrifuge",
            },
            "material_guidance": (
                "The large circular lid insert should be satin silver metal, "
                "not frosted glass."
            ),
        }
    )

    assert "interior wells, trays, liners, rims, recesses" in prompt
    assert "transmission near 1.0 for glass/acrylic" in prompt
    assert (
        "Do not infer a functional viewing window from object category alone" in prompt
    )
    assert "silver metallic lid insert material" in prompt
    assert "User/team material guidance" in prompt
    assert "large circular lid insert should be satin silver metal" in prompt
    assert "Dark Charcoal Matte Interior Plastic" in prompt
    assert "Dark Grey Rubber Seal" in prompt


def test_generated_material_schema_round_trips_and_validates(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="material id cannot be empty"):
        make_material_id(" !!! ")
    with pytest.raises(ValueError, match="USD identifier cannot be empty"):
        make_usd_identifier(" !!! ")

    hints = PBRHints(
        metallic=0.4,
        roughness=0.2,
        opacity=0.75,
        transmission=0.1,
        ior=1.33,
        thin_walled=True,
    )
    assert hints.to_dict() == {
        "metallic": 0.4,
        "roughness": 0.2,
        "opacity": 0.75,
        "transmission": 0.1,
        "ior": 1.33,
        "thin_walled": True,
    }
    assert PBRHints.from_dict(None).roughness == pytest.approx(0.5)
    assert PBRHints.from_dict(hints.to_dict()).thin_walled is True
    for bad_hints in (
        PBRHints(metallic=-0.1),
        PBRHints(roughness=1.1),
        PBRHints(opacity=1.1),
        PBRHints(transmission=-0.1),
        PBRHints(ior=0.0),
    ):
        with pytest.raises(ValueError):
            bad_hints.validate()

    with pytest.raises(TypeError):
        IntendedPart.from_dict("bad")
    part = IntendedPart.from_dict(
        {
            "semantic_label": "outer shell",
            "evidence": "blue",
            "prim_path_hints": "/World/Shell",
        }
    )
    assert part.prim_path_hints == ("/World/Shell",)
    with pytest.raises(ValueError, match="semantic_label"):
        IntendedPart("").validate()

    ref = tmp_path / "refs" / "front.png"
    ref.parent.mkdir()
    ref.write_bytes(b"png")
    recipe = MaterialRecipe.from_dict(
        {
            "id": "blue",
            "name": "Blue Shell",
            "description": "A blue plastic shell",
            "appearance_prompt": "blue plastic",
            "color": "blue",
            "material": "plastic",
            "finish": "glossy",
            "base_color_hint": [0.1, 0.2, 0.3],
            "pbr_hints": hints.to_dict(),
            "reference_image_uris": "refs/front.png",
            "intended_parts": [part.to_dict()],
            "priority": 5,
        },
        base_dir=tmp_path,
    )
    recipe.validate()
    recipe_dict = recipe.to_dict({"albedo": "textures/blue/albedo.png"})
    assert recipe_dict["reference_image_uris"] == [str(ref.resolve())]
    assert recipe_dict["generated_textures"]["albedo"].endswith("albedo.png")
    with pytest.raises(TypeError):
        MaterialRecipe.from_dict([])
    with pytest.raises(ValueError, match="base_color_hint"):
        MaterialRecipe.from_dict(
            {
                "name": "Bad",
                "description": "bad",
                "appearance_prompt": "bad",
                "base_color_hint": [0.1],
            }
        )
    for bad_recipe in (
        MaterialRecipe(
            id="empty_name", name="", description="d", appearance_prompt="p"
        ),
        MaterialRecipe(
            id="empty_description",
            name="n",
            description="",
            appearance_prompt="p",
        ),
        MaterialRecipe(
            id="empty_prompt", name="n", description="d", appearance_prompt=""
        ),
        MaterialRecipe(
            id="bad_color",
            name="n",
            description="d",
            appearance_prompt="p",
            base_color_hint=(1.2, 0.0, 0.0),
        ),
    ):
        with pytest.raises(ValueError):
            bad_recipe.validate()

    plan = MaterialGenerationPlan.from_dict(
        {"version": 3, "asset": {"name": "asset"}, "materials": [recipe_dict]},
        base_dir=tmp_path,
    )
    assert plan.version == 3
    assert plan.to_dict({"blue": {"normal": "textures/blue/normal.png"}})["materials"][
        0
    ]["generated_textures"]["normal"].endswith("normal.png")
    with pytest.raises(TypeError):
        MaterialGenerationPlan.from_dict([])
    with pytest.raises(TypeError):
        MaterialGenerationPlan.from_dict({"materials": {}})
    with pytest.raises(ValueError, match="at least one recipe"):
        MaterialGenerationPlan(materials=()).validate()
    with pytest.raises(ValueError, match="duplicate material name"):
        MaterialGenerationPlan(
            materials=(
                _blue_plastic_recipe(),
                MaterialRecipe(
                    id="different",
                    name=_blue_plastic_recipe().name.upper(),
                    description="d",
                    appearance_prompt="p",
                ),
            )
        ).validate()
    external_ref_recipe = MaterialRecipe.from_dict(
        {
            "id": "external",
            "name": "External",
            "description": "d",
            "appearance_prompt": "p",
            "reference_image_uris": "https://example.com/ref.png",
        },
        base_dir=tmp_path,
    )
    assert external_ref_recipe.reference_image_uris == ("https://example.com/ref.png",)


def test_generated_material_manifest_helpers(tmp_path: Path) -> None:
    texture_dir = tmp_path / "package" / "textures" / "blue"
    texture_dir.mkdir(parents=True)
    texture_map_set = TextureMapSet(
        albedo=texture_dir / "albedo.png",
        normal=texture_dir / "normal.png",
        orm=texture_dir / "orm.png",
    )
    for path in (texture_map_set.albedo, texture_map_set.normal, texture_map_set.orm):
        path.write_bytes(b"png")

    material = GeneratedMaterial(
        recipe=_blue_plastic_recipe(),
        textures=texture_map_set,
        prototype_source={"name": "Prototype", "score": 1.2},
    )
    entry = material_entry(material)
    assert entry["source"] == "generated"
    assert entry["prototype_source"]["name"] == "Prototype"
    assert entry["intended_parts"] == ["outer shell"]
    assert material_entry(material, include_generation_metadata=False) == {
        "name": material.name,
        "description": material.description,
        "binding": material.binding,
    }
    assert material.prototype_name == "Prototype"
    assert (
        GeneratedMaterial(
            recipe=_blue_plastic_recipe(),
            textures=texture_map_set,
        ).prototype_name
        is None
    )
    assert (
        GeneratedMaterial(
            recipe=_blue_plastic_recipe(),
            textures=texture_map_set,
            prototype_source={"name": ""},
        ).prototype_name
        is None
    )

    package_dir = tmp_path / "package"
    library_path = package_dir / "material_library.usda"
    library_path.write_text("#usda 1.0\n")
    manifest_path = write_materials_manifest(
        package_dir / "materials.yaml",
        library_path,
        [material],
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["library_path"] == "material_library.usda"

    plan_path = write_generation_plan(
        package_dir / "generation_plan.yaml",
        MaterialGenerationPlan(materials=(_blue_plastic_recipe(),)),
        [material],
    )
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    assert plan["materials"][0]["generated_textures"]["orm"] == (
        "textures/blue/orm.png"
    )

    library = GeneratedMaterialLibrary(
        package_dir=package_dir,
        material_library_path=library_path,
        materials_manifest_path=manifest_path,
        generation_plan_path=plan_path,
        materials=(material,),
    )
    assert library.materials_data["entries"][0]["prototype_source"]["score"] == 1.2


def test_usd_authoring_pure_material_helpers(tmp_path: Path) -> None:
    texture_dir = tmp_path / "package" / "textures" / "blue"
    texture_dir.mkdir(parents=True)
    texture_map_set = TextureMapSet(
        albedo=texture_dir / "albedo.png",
        normal=texture_dir / "normal.png",
        orm=texture_dir / "orm.png",
    )
    generated = GeneratedMaterial(
        recipe=_blue_plastic_recipe(),
        textures=texture_map_set,
    )
    optical = GeneratedMaterial(
        recipe=MaterialRecipe(
            id="frosted_glass",
            name="Frosted Glass",
            color="clear",
            material="glass",
            finish="milky translucent matte finish",
            description="cloudy frosted glass insert",
            appearance_prompt="milky translucent frosted glass",
            base_color_hint=(-1.0, 0.5, 2.0),
            pbr_hints=PBRHints(
                metallic=0.0,
                roughness=0.2,
                opacity=0.45,
                transmission=0.7,
                ior=1.45,
                thin_walled=True,
            ),
        ),
        textures=texture_map_set,
    )

    assert (
        usd_authoring_module._relative_asset_path(
            texture_map_set.albedo,
            tmp_path / "package" / "material_library.usda",
        )
        == "textures/blue/albedo.png"
    )
    assert "outer shell" in usd_authoring_module._recipe_text(generated)
    assert "housing" in usd_authoring_module._recipe_tokens(generated)
    assert usd_authoring_module._srgb_channel_to_linear(-1.0) == 0.0
    assert usd_authoring_module._srgb_channel_to_linear(2.0) == 1.0
    assert usd_authoring_module._srgb_color_to_linear((0.0, 0.5, 1.0))[0] == 0.0
    assert usd_authoring_module._is_matte(optical) is True
    assert usd_authoring_module._is_metal(generated) is False
    assert (
        usd_authoring_module._is_metal(
            GeneratedMaterial(
                recipe=_brushed_aluminum_recipe(), textures=texture_map_set
            )
        )
        is True
    )
    assert usd_authoring_module._is_optical(optical) is True
    assert usd_authoring_module._optical_transmission(optical) == 0.7
    assert usd_authoring_module._optical_opacity(optical) == 0.45
    assert usd_authoring_module._optical_roughness(optical, 0.2) == 0.5


def test_usd_authoring_copies_prototype_asset_paths(tmp_path: Path) -> None:
    source_dir = tmp_path / "prototype"
    package_dir = tmp_path / "package"
    asset_subdir = Path("prototype_assets") / "blue"
    source_dir.mkdir()
    package_dir.mkdir()
    texture = source_dir / "textures" / "albedo.png"
    texture.parent.mkdir()
    texture.write_bytes(b"png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")

    assert (
        usd_authoring_module._copy_prototype_asset_path(
            "",
            source_dir=source_dir,
            package_dir=package_dir,
            asset_subdir=asset_subdir,
        )
        == ""
    )
    assert (
        usd_authoring_module._copy_prototype_asset_path(
            "https://example.com/texture.png",
            source_dir=source_dir,
            package_dir=package_dir,
            asset_subdir=asset_subdir,
        )
        == ""
    )
    assert (
        usd_authoring_module._copy_prototype_asset_path(
            str(outside),
            source_dir=source_dir,
            package_dir=package_dir,
            asset_subdir=asset_subdir,
        )
        == ""
    )
    assert (
        usd_authoring_module._copy_prototype_asset_path(
            "../outside.png",
            source_dir=source_dir,
            package_dir=package_dir,
            asset_subdir=asset_subdir,
        )
        == ""
    )
    assert (
        usd_authoring_module._copy_prototype_asset_path(
            "missing.png",
            source_dir=source_dir,
            package_dir=package_dir,
            asset_subdir=asset_subdir,
        )
        == ""
    )

    copied_relative = usd_authoring_module._copy_prototype_asset_path(
        "textures/albedo.png",
        source_dir=source_dir,
        package_dir=package_dir,
        asset_subdir=asset_subdir,
    )
    assert copied_relative == "prototype_assets/blue/textures/albedo.png"
    assert (package_dir / copied_relative).read_bytes() == b"png"


def test_usd_authoring_layer_and_fake_input_helpers(tmp_path: Path) -> None:
    layer = Sdf.Layer.CreateAnonymous()
    usd_authoring_module._ensure_parent_specs(
        layer,
        Sdf.Path("/World/Looks/Generated_Blue"),
        Sdf,
    )
    assert layer.GetPrimAtPath("/World")
    assert layer.GetPrimAtPath("/World/Looks")

    class FakeAttr:
        def __init__(self) -> None:
            self.color_space = None

        def SetColorSpace(self, value: str) -> None:
            self.color_space = value

    class FakeInput:
        def __init__(self) -> None:
            self.value = None
            self.attr = FakeAttr()

        def Set(self, value) -> None:
            self.value = value

        def GetAttr(self) -> FakeAttr:
            return self.attr

    class FakeMaterial:
        def __init__(self) -> None:
            self.inputs = {"roughness": FakeInput()}
            self.created: list[tuple[str, object]] = []

        def GetInput(self, name: str):
            return self.inputs.get(name)

        def CreateInput(self, name: str, value_type):
            self.created.append((name, value_type))
            self.inputs[name] = FakeInput()
            return self.inputs[name]

    fake_sdf = type(
        "FakeSdf",
        (),
        {
            "AssetPath": lambda value: f"asset:{value}",
            "ValueTypeNames": type("ValueTypeNames", (), {"Asset": "asset"}),
        },
    )
    material = FakeMaterial()
    usd_authoring_module._set_existing_material_input(material, "roughness", 0.4)
    usd_authoring_module._set_existing_material_input(material, "missing", 1.0)
    assert material.inputs["roughness"].value == 0.4

    usd_authoring_module._set_existing_asset_material_input(
        material,
        "albedo_file",
        "textures/albedo.png",
        color_space="auto",
        create_missing=True,
        sdf=fake_sdf,
    )
    assert material.created == [("albedo_file", "asset")]
    assert material.inputs["albedo_file"].value == "textures/albedo.png"
    assert material.inputs["albedo_file"].attr.color_space == "auto"
    usd_authoring_module._set_existing_asset_material_input(
        material,
        "ignored",
        "x.png",
    )
    assert "ignored" not in material.inputs
    usd_authoring_module._clear_existing_asset_material_input(
        material,
        "albedo_file",
        fake_sdf,
    )
    assert material.inputs["albedo_file"].value == "asset:"


def test_usd_authoring_copies_asset_arrays_into_package(tmp_path: Path) -> None:
    source_dir = tmp_path / "prototype"
    package_dir = tmp_path / "package"
    source_dir.mkdir()
    package_dir.mkdir()
    texture = source_dir / "textures" / "albedo.png"
    texture.parent.mkdir()
    texture.write_bytes(b"png")

    class FakeAssetPath:
        def __init__(self, path: str) -> None:
            self.path = path

        def __eq__(self, other) -> bool:
            return isinstance(other, FakeAssetPath) and self.path == other.path

    class FakeAssetPathArray(list):
        pass

    class FakeSdf:
        AssetPath = FakeAssetPath
        AssetPathArray = FakeAssetPathArray

    class FakeAttrSpec:
        def __init__(self, default) -> None:
            self.default = default

    class FakeChild:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakePath(str):
        def AppendChild(self, name: str):
            return FakePath(f"{self}/{name}")

    class FakePrimSpec:
        def __init__(self, *, child: bool = False) -> None:
            self.attributes = {
                "single": FakeAttrSpec(FakeAssetPath("textures/albedo.png")),
                "array": FakeAttrSpec(
                    FakeAssetPathArray([FakeAssetPath("textures/albedo.png")])
                ),
            }
            self.nameChildren = [] if child else [FakeChild("Child")]

    class FakeLayer:
        def __init__(self) -> None:
            self.root = FakePrimSpec()
            self.child = FakePrimSpec(child=True)

        def GetPrimAtPath(self, path):
            if str(path).endswith("/Child"):
                return self.child
            return self.root

    layer = FakeLayer()
    empty_layer = type("EmptyLayer", (), {"GetPrimAtPath": lambda self, path: None})()
    usd_authoring_module._copy_prototype_assets_into_package(
        empty_layer,
        FakePath("/World/Looks/Missing"),
        source_dir=source_dir,
        package_dir=package_dir,
        asset_subdir=Path("prototype_assets") / "blue",
        Sdf=FakeSdf,
    )
    usd_authoring_module._copy_prototype_assets_into_package(
        layer,
        FakePath("/World/Looks/Prototype"),
        source_dir=source_dir,
        package_dir=package_dir,
        asset_subdir=Path("prototype_assets") / "blue",
        Sdf=FakeSdf,
    )

    assert layer.root.attributes["single"].default.path == (
        "prototype_assets/blue/textures/albedo.png"
    )
    assert layer.root.attributes["array"].default[0].path == (
        "prototype_assets/blue/textures/albedo.png"
    )


def test_usd_authoring_try_author_from_prototype_with_real_sdf_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    texture_map_set = TextureMapSet(
        albedo=tmp_path / "albedo.png",
        normal=tmp_path / "normal.png",
        orm=tmp_path / "orm.png",
    )
    source_path = tmp_path / "prototype.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_path))
    Sdf.CreatePrimInLayer(source_layer, Sdf.Path("/World/Looks/Prototype"))
    source_layer.Save()
    library_path = tmp_path / "package" / "material_library.usda"
    library_path.parent.mkdir()
    stage = Usd.Stage.CreateNew(str(library_path))
    adapted: list[tuple[str, str]] = []

    generated = GeneratedMaterial(
        recipe=_blue_plastic_recipe(),
        textures=texture_map_set,
        prototype_source={
            "library_path": str(source_path),
            "binding": "/World/Looks/Prototype",
        },
    )

    monkeypatch.setattr(
        usd_authoring_module,
        "_adapt_openpbr_material",
        lambda stage, binding, material, library, sdf: adapted.append(
            (binding, str(library))
        ),
    )
    import material_agent.tasks.apply_materials_to_usd as apply_materials_module

    monkeypatch.setattr(
        apply_materials_module,
        "clear_color_space_on_empty_asset_inputs",
        lambda layer, target_path: None,
    )

    assert (
        usd_authoring_module._try_author_from_prototype(
            stage,
            library_path,
            GeneratedMaterial(recipe=_blue_plastic_recipe(), textures=texture_map_set),
        )
        is False
    )
    assert (
        usd_authoring_module._try_author_from_prototype(stage, library_path, generated)
        is True
    )
    assert adapted == [(generated.binding, str(library_path))]


def test_usd_authoring_shader_adaptation_helpers(tmp_path: Path) -> None:
    library_path = tmp_path / "library.usda"
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    texture_map_set = TextureMapSet(
        albedo=texture_dir / "albedo.png",
        normal=texture_dir / "normal.png",
        orm=texture_dir / "orm.png",
    )
    for path in (texture_map_set.albedo, texture_map_set.normal, texture_map_set.orm):
        path.write_bytes(b"png")

    def make_stage_with_material(path: str):
        stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(stage, path)
        for name, value_type in {
            "base_color": Sdf.ValueTypeNames.Color3f,
            "base_metalness": Sdf.ValueTypeNames.Float,
            "specular_roughness": Sdf.ValueTypeNames.Float,
            "roughness": Sdf.ValueTypeNames.Float,
            "transmission_weight": Sdf.ValueTypeNames.Float,
            "transmission_color": Sdf.ValueTypeNames.Color3f,
            "geometry_opacity": Sdf.ValueTypeNames.Float,
            "geometry_thin_walled": Sdf.ValueTypeNames.Bool,
            "transmission_weight_texture_file": Sdf.ValueTypeNames.Asset,
            "transmission_color_texture_file": Sdf.ValueTypeNames.Asset,
            "geometry_opacity_texture_file": Sdf.ValueTypeNames.Asset,
            "specular_ior": Sdf.ValueTypeNames.Float,
            "specular_weight": Sdf.ValueTypeNames.Float,
            "coat_weight": Sdf.ValueTypeNames.Float,
            "coat_roughness": Sdf.ValueTypeNames.Float,
        }.items():
            material.CreateInput(name, value_type)

        preview = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        for name, value_type in {
            "diffuseColor": Sdf.ValueTypeNames.Color3f,
            "metallic": Sdf.ValueTypeNames.Float,
            "roughness": Sdf.ValueTypeNames.Float,
            "transmission_weight": Sdf.ValueTypeNames.Float,
            "transmission_color": Sdf.ValueTypeNames.Color3f,
            "geometry_opacity": Sdf.ValueTypeNames.Float,
        }.items():
            preview.CreateInput(name, value_type)

        albedo = UsdShade.Shader.Define(stage, f"{path}/AlbedoTexture")
        albedo.CreateIdAttr("UsdUVTexture")
        albedo.CreateInput("file", Sdf.ValueTypeNames.Asset)
        albedo.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token)
        return stage, material

    generated = GeneratedMaterial(
        recipe=_blue_plastic_recipe(), textures=texture_map_set
    )
    stage, material = make_stage_with_material(generated.binding)
    usd_authoring_module._adapt_openpbr_material(
        stage,
        generated.binding,
        generated,
        library_path,
        Sdf,
    )
    assert material.GetInput("base_color_texture_file").Get() == Sdf.AssetPath(
        "textures/albedo.png"
    )
    assert material.GetInput("transmission_weight").Get() == 0.0
    albedo_shader = UsdShade.Shader(
        stage.GetPrimAtPath(f"{generated.binding}/AlbedoTexture")
    )
    assert albedo_shader.GetInput("file").Get() == Sdf.AssetPath("textures/albedo.png")
    assert albedo_shader.GetInput("sourceColorSpace").Get() == "auto"

    optical_recipe = MaterialRecipe(
        id="frosted_glass",
        name="Frosted Glass",
        color="clear",
        material="glass",
        finish="frosted matte",
        description="frosted translucent glass insert",
        appearance_prompt="frosted translucent glass",
        base_color_hint=(0.7, 0.8, 0.9),
        pbr_hints=PBRHints(
            roughness=0.2,
            opacity=0.4,
            transmission=0.6,
            ior=1.4,
            thin_walled=True,
        ),
    )
    optical = GeneratedMaterial(recipe=optical_recipe, textures=texture_map_set)
    optical_stage, optical_material = make_stage_with_material(optical.binding)
    usd_authoring_module._adapt_openpbr_material(
        optical_stage,
        optical.binding,
        optical,
        library_path,
        Sdf,
    )
    assert optical_material.GetInput("transmission_weight").Get() == pytest.approx(0.6)
    assert optical_material.GetInput("geometry_opacity").Get() == pytest.approx(0.4)
    assert optical_material.GetInput("geometry_thin_walled").Get() is True
    assert optical_material.GetInput("coat_weight").Get() == 0.0


def test_usd_authoring_profile_wrappers_with_fake_usdex(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "library.usda"
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    texture_map_set = TextureMapSet(
        albedo=texture_dir / "albedo.png",
        normal=texture_dir / "normal.png",
        orm=texture_dir / "orm.png",
    )
    for path in (texture_map_set.albedo, texture_map_set.normal, texture_map_set.orm):
        path.write_bytes(b"png")
    generated = GeneratedMaterial(
        recipe=_blue_plastic_recipe(), textures=texture_map_set
    )

    fake_usdex = types.ModuleType("usdex")
    fake_core = types.ModuleType("usdex.core")
    fake_rtx = types.ModuleType("usdex.rtx")
    fake_usdex.core = fake_core
    fake_usdex.rtx = fake_rtx
    monkeypatch.setitem(sys.modules, "usdex", fake_usdex)
    monkeypatch.setitem(sys.modules, "usdex.core", fake_core)
    monkeypatch.setitem(sys.modules, "usdex.rtx", fake_rtx)

    stage = Usd.Stage.CreateInMemory()

    def fake_define_preview_material(stage, path, color, opacity, roughness, metallic):
        material = UsdShade.Material.Define(stage, str(path))
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        return material

    fake_core.definePreviewMaterial = fake_define_preview_material
    assert usd_authoring_module.can_author_openpbr_materialx_with_usdex() is False
    preview_material = usd_authoring_module._define_preview_material_from_recipe(
        stage,
        library_path,
        generated,
        generated.binding,
        0.3,
        Sdf,
    )
    assert preview_material.GetPrim().GetPath() == Sdf.Path(generated.binding)
    preview_shader = UsdShade.Shader.Get(
        stage,
        f"{generated.binding}/PreviewSurface",
    )
    assert preview_shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert preview_shader.GetInput("roughness").Get() == pytest.approx(0.3)
    assert preview_shader.GetInput("metallic").Get() == pytest.approx(
        generated.recipe.pbr_hints.metallic
    )
    assert preview_shader.GetInput("opacity").Get() == pytest.approx(1.0)
    assert preview_material.GetSurfaceOutput().HasConnectedSource()

    openpbr_calls: list[tuple[str, object]] = []
    fake_core.definePbrMaterial = fake_define_preview_material
    fake_core.addDiffuseTextureToPbrMaterial = (
        lambda material, path: openpbr_calls.append(("diffuse", path))
    )
    fake_core.addNormalTextureToPbrMaterial = (
        lambda material, path: openpbr_calls.append(("normal", path))
    )
    fake_core.addOrmTextureToPbrMaterial = lambda material, path: openpbr_calls.append(
        ("orm", path)
    )
    assert usd_authoring_module.can_author_openpbr_materialx_with_usdex() is True
    openpbr_material = usd_authoring_module._define_openpbr_materialx_from_recipe(
        stage,
        library_path,
        generated,
        "/World/Looks/OpenPBR",
        0.4,
        Sdf,
    )
    assert openpbr_material.GetPrim().GetPath() == Sdf.Path("/World/Looks/OpenPBR")
    assert [call[0] for call in openpbr_calls] == ["diffuse", "normal", "orm"]

    del fake_core.definePbrMaterial
    with pytest.raises(
        usd_authoring_module.MaterialAuthoringPrerequisiteError,
        match="definePbrMaterial",
    ):
        usd_authoring_module.require_material_authoring_prerequisites(
            "openpbr_materialx"
        )

    rtx_calls: list[tuple[str, object]] = []

    def fake_define_rtx_material(stage, path, color, opacity, roughness, metallic):
        material = UsdShade.Material.Define(stage, str(path))
        rtx_calls.append(("define", str(path)))
        return material

    fake_rtx.definePbrMaterial = fake_define_rtx_material
    fake_rtx.addDiffuseTextureToPbrMaterial = lambda material, path: rtx_calls.append(
        ("diffuse", path)
    )
    fake_rtx.addNormalTextureToPbrMaterial = lambda material, path: rtx_calls.append(
        ("normal", path)
    )
    fake_rtx.addOrmTextureToPbrMaterial = lambda material, path: rtx_calls.append(
        ("orm", path)
    )
    omnipbr = usd_authoring_module._define_omnipbr_mdl_material_from_recipe(
        stage,
        library_path,
        generated,
        "/World/Looks/OmniPBR",
        0.5,
        Sdf,
    )
    assert omnipbr.GetPrim().GetPath() == Sdf.Path("/World/Looks/OmniPBR")
    assert [call[0] for call in rtx_calls] == ["define", "diffuse", "normal", "orm"]

    connected = UsdShade.Material.Define(stage, "/World/Looks/Connected")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Connected/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    connected.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    usd_authoring_module._connect_preview_material_surface(
        connected, Sdf, Usd, UsdShade
    )

    needs_connection = UsdShade.Material.Define(stage, "/World/Looks/NeedsConnection")
    shader = UsdShade.Shader.Define(
        stage,
        "/World/Looks/NeedsConnection/PreviewSurface",
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    usd_authoring_module._connect_preview_material_surface(
        needs_connection,
        Sdf,
        Usd,
        UsdShade,
    )
    assert needs_connection.GetSurfaceOutput().HasConnectedSource()

    unconnected = UsdShade.Material.Define(stage, "/World/Looks/Unconnected")
    UsdShade.Shader.Define(stage, "/World/Looks/Unconnected/OtherShader")
    with pytest.raises(ValueError, match="UsdPreviewSurface"):
        usd_authoring_module._connect_preview_material_surface(
            unconnected,
            Sdf,
            Usd,
            UsdShade,
        )


def test_usd_authoring_remaining_guard_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    texture_map_set = TextureMapSet(
        albedo=texture_dir / "albedo.png",
        normal=texture_dir / "normal.png",
        orm=texture_dir / "orm.png",
    )
    for path in (texture_map_set.albedo, texture_map_set.normal, texture_map_set.orm):
        path.write_bytes(b"png")

    glass_recipe = MaterialRecipe(
        id="clear_glass_default",
        name="Clear Glass Default",
        color="clear",
        material="glass",
        finish="glossy",
        description="clear glass lens",
        appearance_prompt="clear glass lens",
        base_color_hint=(0.8, 0.9, 1.0),
    )
    glass = GeneratedMaterial(recipe=glass_recipe, textures=texture_map_set)
    assert usd_authoring_module._is_optical(glass) is True
    assert usd_authoring_module._optical_transmission(glass) == 1.0

    source_dir = tmp_path / "source"
    package_dir = tmp_path / "package"
    source_dir.mkdir()
    package_dir.mkdir()
    outside_asset = tmp_path / "outside.png"
    outside_asset.write_bytes(b"asset")
    monkeypatch.setattr(usd_authoring_module, "is_relative_to", lambda *_args: True)
    assert (
        usd_authoring_module._copy_prototype_asset_path(
            str(outside_asset),
            source_dir=source_dir,
            package_dir=package_dir,
            asset_subdir=Path("assets"),
        )
        == ""
    )

    stage = Usd.Stage.CreateInMemory()
    missing_layer = GeneratedMaterial(
        recipe=_blue_plastic_recipe(),
        textures=texture_map_set,
        prototype_source={
            "library_path": tmp_path / "missing.usda",
            "binding": "/World/Looks/Source",
        },
    )
    assert (
        usd_authoring_module._try_author_from_prototype(
            stage,
            tmp_path / "library.usda",
            missing_layer,
        )
        is False
    )

    empty_layer_path = tmp_path / "empty.usda"
    empty_layer = Sdf.Layer.CreateNew(str(empty_layer_path))
    empty_layer.Save()
    missing_binding = GeneratedMaterial(
        recipe=_blue_plastic_recipe(),
        textures=texture_map_set,
        prototype_source={
            "library_path": empty_layer_path,
            "binding": "/World/Looks/Missing",
        },
    )
    assert (
        usd_authoring_module._try_author_from_prototype(
            stage,
            tmp_path / "library.usda",
            missing_binding,
        )
        is False
    )

    source_layer_path = tmp_path / "source.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_layer_path))
    Sdf.CreatePrimInLayer(source_layer, Sdf.Path("/World/Looks/Source"))
    source_layer.Save()
    copy_fails = GeneratedMaterial(
        recipe=_blue_plastic_recipe(),
        textures=texture_map_set,
        prototype_source={
            "library_path": source_layer_path,
            "binding": "/World/Looks/Source",
        },
    )
    monkeypatch.setattr(Sdf, "CopySpec", lambda *_args: False)
    assert (
        usd_authoring_module._try_author_from_prototype(
            stage,
            tmp_path / "library.usda",
            copy_fails,
        )
        is False
    )

    real_import_module = usd_authoring_module.importlib.import_module

    def fake_import_module(name: str):
        if name == "usdex.core":
            raise ImportError("usdex is not installed")
        return real_import_module(name)

    monkeypatch.delitem(sys.modules, "usdex", raising=False)
    monkeypatch.delitem(sys.modules, "usdex.core", raising=False)
    monkeypatch.setattr(
        usd_authoring_module.importlib,
        "import_module",
        fake_import_module,
    )
    assert usd_authoring_module.can_author_openpbr_materialx_with_usdex() is False

    class FakeChild:
        def GetName(self) -> str:
            return "AlbedoTexture"

    class FakePrim:
        def GetChildren(self) -> list[FakeChild]:
            return [FakeChild()]

    monkeypatch.setattr(UsdShade, "Shader", lambda _child: None)
    usd_authoring_module._set_shader_input_if_present(FakePrim(), "roughness", 0.25)
    usd_authoring_module._set_albedo_asset_input_if_present(
        FakePrim(),
        "textures/albedo.png",
        Sdf,
    )


def test_write_material_library_usd_with_stubbed_usdex_authoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    texture_dir = tmp_path / "textures" / "blue"
    texture_dir.mkdir(parents=True)
    texture_map_set = TextureMapSet(
        albedo=texture_dir / "albedo.png",
        normal=texture_dir / "normal.png",
        orm=texture_dir / "orm.png",
    )
    for path in (texture_map_set.albedo, texture_map_set.normal, texture_map_set.orm):
        path.write_bytes(b"png")
    generated = GeneratedMaterial(
        recipe=_blue_plastic_recipe(), textures=texture_map_set
    )

    import world_understanding.utils.usd.material as usd_material_module

    fallback_calls = []
    monkeypatch.setattr(
        usd_material_module,
        "add_ovrtx_preview_fallbacks_for_materialx_openpbr",
        lambda stage: fallback_calls.append(stage),
    )

    def fake_define_preview_material(
        stage,
        library_path,
        generated,
        material_path,
        roughness,
        Sdf,
    ):
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        return material

    branch_calls: list[tuple[str, str, float]] = []

    def fake_define_profile_material(
        stage,
        library_path,
        generated,
        material_path,
        roughness,
        Sdf,
    ):
        branch_calls.append((str(library_path), material_path, roughness))
        return UsdShade.Material.Define(stage, material_path)

    monkeypatch.setattr(
        usd_authoring_module,
        "_define_preview_material_from_recipe",
        fake_define_preview_material,
    )
    monkeypatch.setattr(
        usd_authoring_module,
        "_define_openpbr_materialx_from_recipe",
        fake_define_profile_material,
    )
    monkeypatch.setattr(
        usd_authoring_module,
        "_define_omnipbr_mdl_material_from_recipe",
        fake_define_profile_material,
    )
    monkeypatch.setattr(
        usd_authoring_module,
        "require_material_authoring_prerequisites",
        lambda material_profile: None,
    )
    monkeypatch.setattr(
        usd_authoring_module,
        "inspect_material_library_authoring",
        lambda library_path, materials, *, material_profile: {},
    )

    with pytest.raises(ValueError, match="at least one generated material"):
        usd_authoring_module.write_material_library_usd(tmp_path / "empty.usda", [])
    with pytest.raises(ValueError, match="display_color"):
        usd_authoring_module.write_material_library_usd(
            tmp_path / "display.usda",
            [generated],
            material_profile="display_color",
        )

    authored = usd_authoring_module.write_material_library_usd(
        tmp_path / "preview.usda",
        [generated],
    )
    assert authored.exists()
    stage = Usd.Stage.Open(str(authored))
    assert stage.GetPrimAtPath(f"{generated.binding}/AlbedoTexture")
    assert stage.GetPrimAtPath(f"{generated.binding}/NormalTexture")
    assert stage.GetPrimAtPath(f"{generated.binding}/OrmTexture")
    assert fallback_calls

    usd_authoring_module.write_material_library_usd(
        tmp_path / "openpbr.usda",
        [generated],
        material_profile="openpbr_materialx",
    )
    usd_authoring_module.write_material_library_usd(
        tmp_path / "omnipbr.usda",
        [generated],
        material_profile="omnipbr_mdl",
    )
    assert [call[1] for call in branch_calls] == [generated.binding, generated.binding]

    monkeypatch.setattr(
        usd_authoring_module,
        "_try_author_from_prototype",
        lambda stage, library_path, generated: True,
    )
    prototype_only = usd_authoring_module.write_material_library_usd(
        tmp_path / "prototype-only.usda",
        [generated],
    )
    assert prototype_only.exists()


def test_generated_material_prototype_helpers(tmp_path: Path) -> None:
    assert load_material_prototypes_from_data(None) == ()
    assert load_material_prototypes_from_data({"entries": []}) == ()
    assert load_material_prototypes_from_data({"library_path": "missing.usda"}) == ()

    data = {
        "library_path": "library.usda",
        "entries": [
            "bad",
            {"name": "", "binding": "/World/Looks/Bad"},
            {"name": "Missing Binding"},
            {
                "name": "Blue Glossy Plastic",
                "description": "saturated blue glossy plastic body shell",
                "binding": "/World/Looks/Blue",
            },
            {
                "name": "Silver Metal",
                "description": "silver aluminum polished metal",
                "binding": "/World/Looks/Silver",
            },
            {
                "name": "Clear Glass",
                "description": "frosted translucent glass acrylic",
                "binding": "/World/Looks/Glass",
            },
        ],
    }
    prototypes = load_material_prototypes_from_data(data, base_dir=tmp_path)
    assert [prototype.name for prototype in prototypes] == [
        "Blue Glossy Plastic",
        "Silver Metal",
        "Clear Glass",
    ]
    assert prototypes[0].to_source_dict(score=1.5)["score"] == 1.5

    manifest_path = tmp_path / "materials.yaml"
    manifest_path.write_text("entries: []\n", encoding="utf-8")
    assert load_material_prototypes_from_manifest(manifest_path) == ()

    blue_score = score_material_prototype(_blue_plastic_recipe(), prototypes[0])
    glass_score = score_material_prototype(_blue_plastic_recipe(), prototypes[2])
    assert blue_score > glass_score
    assert (
        select_material_prototype(_blue_plastic_recipe(), prototypes, min_score=0.0)[
            0
        ].name
        == "Blue Glossy Plastic"
    )
    assert select_material_prototype(_blue_plastic_recipe(), [], min_score=0.0) is None
    assert (
        select_material_prototype(_blue_plastic_recipe(), prototypes, min_score=999.0)
        is None
    )

    metallic = _brushed_aluminum_recipe()
    assert score_material_prototype(metallic, prototypes[1]) > score_material_prototype(
        metallic, prototypes[0]
    )
    optical = _frosted_glass_recipe()
    assert score_material_prototype(optical, prototypes[2]) > score_material_prototype(
        optical, prototypes[1]
    )

    no_token_recipe = MaterialRecipe(
        id="plain",
        name="",
        description="",
        appearance_prompt="",
    )
    assert score_material_prototype(no_token_recipe, prototypes[0]) == 0.0
    assert (
        score_material_prototype(
            _blue_plastic_recipe(),
            MaterialPrototype(
                name="",
                description="",
                binding="/World/Looks/X",
                library_path=Path("x.usda"),
            ),
        )
        == 0.0
    )


def test_generated_material_prototype_helpers_read_stage_values(
    tmp_path: Path,
) -> None:
    library_path = tmp_path / "prototype_library.usda"
    stage = Usd.Stage.CreateNew(str(library_path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    material.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.02, 0.15, 0.85)
    )
    material.CreateInput("base_metalness", Sdf.ValueTypeNames.String).Set("bad")
    child_material = UsdShade.Material.Define(stage, "/World/Looks/Child")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Child/Shader")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.5, 0.6, 0.7)
    )
    child_material.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(0.2)
    empty_material = UsdShade.Material.Define(stage, "/World/Looks/Empty")
    UsdGeom.Scope.Define(stage, "/World/Looks/Empty/NonShaderChild")
    UsdShade.Shader.Define(stage, "/World/Looks/Empty/Shader")
    stage.GetRootLayer().Save()

    prototypes = load_material_prototypes_from_data(
        {
            "library_path": "prototype_library.usda",
            "entries": [
                {
                    "name": "Blue Plastic",
                    "description": "blue plastic",
                    "binding": "/World/Looks/Blue",
                },
                {
                    "name": "Child Shader",
                    "description": "gray plastic",
                    "binding": "/World/Looks/Child",
                },
            ],
        },
        base_dir=tmp_path,
    )

    assert prototypes[0].base_color == pytest.approx((0.02, 0.15, 0.85))
    assert prototypes[0].metalness is None
    assert prototypes[1].base_color == pytest.approx((0.5, 0.6, 0.7))
    assert prototypes[1].metalness == pytest.approx(0.2)
    assert _read_material_input(stage, "/World/Looks/Missing", "base_color") is None
    assert (
        _read_material_input(stage, str(empty_material.GetPath()), "base_color") is None
    )
    assert _as_color(None) is None
    assert _as_color(1.0) is None
    assert _as_color([1.0, 0.5]) is None
    assert _color_similarity((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == 0.0


def test_generated_material_prototype_scoring_edges() -> None:
    assert (
        score_material_prototype(
            _brushed_aluminum_recipe(),
            MaterialPrototype(
                name="Polished Silver Metal",
                description="steel metal",
                binding="/World/Looks/Metal",
                library_path=Path("metal.usda"),
                base_color=(0.7, 0.7, 0.7),
                metalness=0.8,
            ),
        )
        > 0.0
    )

    assert (
        score_material_prototype(
            _blue_plastic_recipe(),
            MaterialPrototype(
                name="Car Paint Blue",
                description="blue painted car shell",
                binding="/World/Looks/CarPaint",
                library_path=Path("car.usda"),
                metalness=0.1,
            ),
        )
        > 0.0
    )

    rubber_score = score_material_prototype(
        _blue_plastic_recipe(),
        MaterialPrototype(
            name="Rubber Black",
            description="black rubber",
            binding="/World/Looks/Rubber",
            library_path=Path("rubber.usda"),
        ),
    )
    assert rubber_score < score_material_prototype(
        _blue_plastic_recipe(),
        MaterialPrototype(
            name="Blue Plastic",
            description="blue plastic",
            binding="/World/Looks/BluePlastic",
            library_path=Path("plastic.usda"),
        ),
    )


def test_generated_material_validation_error_paths(tmp_path: Path) -> None:
    missing = validate_generated_material_library(tmp_path / "missing.yaml")
    assert missing.errors[0].startswith("materials manifest not found")

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":", encoding="utf-8")
    assert "failed to parse" in validate_generated_material_library(bad_yaml).errors[0]

    no_library = tmp_path / "no-library.yaml"
    no_library.write_text("entries: []\n", encoding="utf-8")
    assert validate_generated_material_library(no_library).errors == (
        "materials manifest is missing library_path",
    )

    missing_library = tmp_path / "missing-library.yaml"
    missing_library.write_text("library_path: missing.usda\n", encoding="utf-8")
    assert (
        "material library USD not found"
        in validate_generated_material_library(missing_library).errors[0]
    )

    library_path = tmp_path / "library.usda"
    stage = Usd.Stage.CreateNew(str(library_path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    material.CreateInput("albedo", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/blue.png")
    )
    material.CreateInput("external", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("https://example.com/remote.png")
    )
    stage.GetRootLayer().Save()

    manifest_path = tmp_path / "materials.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "library_path": "library.usda",
                "entries": [
                    "bad",
                    {"name": "No Binding"},
                    {"name": "Missing Prim", "binding": "/World/Looks/Missing"},
                    {"name": "Blue", "binding": "/World/Looks/Blue"},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = validate_generated_material_library(manifest_path)
    assert not result.ok
    assert "material entry 0 must be a mapping" in result.errors
    assert "material entry 'No Binding' is missing binding" in result.errors
    assert any("Missing Prim" in error for error in result.errors)
    assert any("texture is missing" in error for error in result.errors)
    assert result.warnings == (
        "non-relative asset path in material library: https://example.com/remote.png",
    )
    assert result.metadata["entry_count"] == 4
    assert result.metadata["texture_count"] == 1


def test_generated_material_validation_additional_edge_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    (texture_dir / "not_png.txt").write_text("not png", encoding="utf-8")
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(texture_dir / "transparent.png")
    Image.new("RGB", (2, 2), (10, 20, 30)).save(texture_dir / "ok.png")

    library_path = tmp_path / "library.usda"
    stage = Usd.Stage.CreateNew(str(library_path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    material.CreateInput("unset", Sdf.ValueTypeNames.Asset)
    material.CreateInput("bad_type", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/not_png.txt")
    )
    material.CreateInput("transparent", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/transparent.png")
    )
    material.CreateInput("outside", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../outside.png")
    )
    material.CreateInput("ok", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/ok.png")
    )
    stage.GetRootLayer().Save()

    manifest_path = tmp_path / "materials.yaml"
    manifest_path.write_text(
        yaml.safe_dump({"library_path": "library.usda", "entries": []}),
        encoding="utf-8",
    )

    result = validate_generated_material_library(manifest_path)

    assert "materials manifest has no entries" in result.errors
    assert any("not_png.txt" in error for error in result.errors)
    assert any("transparent.png" in error for error in result.errors)
    assert any("escapes material library package" in error for error in result.errors)
    assert result.metadata["texture_count"] == 3

    monkeypatch.setattr(Usd.Stage, "Open", lambda *args, **kwargs: None)
    failed_open = validate_generated_material_library(manifest_path)
    assert failed_open.errors == (
        f"failed to open material library USD: {library_path}",
    )


def test_generated_material_texture_generation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _blue_plastic_recipe()
    with pytest.raises(ValueError, match="texture_size"):
        generate_texture_maps(
            recipe,
            tmp_path / "bad-size",
            settings=TextureGenerationSettings(texture_size=0),
        )
    with pytest.raises(ValueError, match="albedo_color_correction_strength"):
        generate_texture_maps(
            recipe,
            tmp_path / "bad-strength",
            settings=TextureGenerationSettings(albedo_color_correction_strength=1.5),
        )

    assert _deterministic_span_value(1, 0, 0, 5, 5) == 5
    image = Image.new("RGB", (2, 2), (10, 20, 30))
    assert _match_albedo_mean_to_base_color(image, (1, 1, 1), 0).getpixel((0, 0)) == (
        10,
        20,
        30,
    )

    ref = tmp_path / "ref.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(ref)

    class ResizingModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int | None]] = []

        def generate(self, prompt: str, images=None):
            self.calls.append((prompt, len(images) if images else None))
            return Image.new("RGB", (2, 2), (20, 30, 40))

    model = ResizingModel()
    maps = generate_texture_maps(
        MaterialRecipe(
            name="Ref Material",
            description="desc",
            appearance_prompt="prompt",
            reference_image_uris=(str(ref),),
        ),
        tmp_path / "model",
        settings=TextureGenerationSettings(
            texture_size=4,
            color_correct_albedo=False,
        ),
        image_model=model,
    )
    assert Image.open(maps.albedo).size == (4, 4)
    assert model.calls[0][1] == 1

    class FlakyModel:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return None
            return Image.new("RGB", (4, 4), (60, 70, 80))

    flaky = FlakyModel()
    maps = generate_texture_maps(
        recipe,
        tmp_path / "flaky",
        settings=TextureGenerationSettings(texture_size=4, color_correct_albedo=False),
        image_model=flaky,
    )
    assert flaky.calls == 2
    assert Image.open(maps.albedo).getpixel((0, 0)) == (60, 70, 80)

    class FailingModel:
        def generate(self, *args, **kwargs):
            raise RuntimeError("no image")

    maps = generate_texture_maps(
        recipe,
        tmp_path / "fallback",
        settings=TextureGenerationSettings(texture_size=4),
        image_model=FailingModel(),
    )
    assert maps.albedo.exists()

    monkeypatch.setenv("IMAGE_BACKEND_KEY", "secret")
    observed: dict[str, object] = {}

    def fake_create_image_generation_model(backend: str, **kwargs: object) -> object:
        observed["backend"] = backend
        observed["kwargs"] = kwargs
        return object()

    import world_understanding.functions.models.image_generation_models as image_models

    monkeypatch.setattr(
        image_models,
        "create_image_generation_model",
        fake_create_image_generation_model,
    )
    for reference in ("IMAGE_BACKEND_KEY", "${IMAGE_BACKEND_KEY}"):
        observed.clear()
        model = _create_image_generation_model(
            TextureGenerationSettings(
                backend="mock",
                model="m",
                base_url="http://local",
                api_key_env_var=reference,
            )
        )
        assert model is not None
        assert observed == {
            "backend": "mock",
            "kwargs": {
                "model": "m",
                "base_url": "http://local",
                "api_key": "secret",
            },
        }


def test_build_generated_material_library_orchestrates_package_without_native_authoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, TextureGenerationSettings | None, object | None]] = []
    authored: list[tuple[Path, tuple[GeneratedMaterial, ...], str]] = []

    def fake_generate_texture_maps(
        recipe: MaterialRecipe,
        output_dir: Path,
        *,
        settings: TextureGenerationSettings | None,
        image_model: object | None,
    ) -> TextureMapSet:
        output_dir.mkdir(parents=True)
        maps = TextureMapSet(
            albedo=output_dir / "albedo.png",
            normal=output_dir / "normal.png",
            orm=output_dir / "orm.png",
        )
        for path in (maps.albedo, maps.normal, maps.orm):
            Image.new("RGB", (2, 2), (10, 20, 30)).save(path)
        calls.append((recipe.material_id, settings, image_model))
        return maps

    def fake_write_material_library_usd(
        material_library_path: Path,
        materials: tuple[GeneratedMaterial, ...],
        *,
        material_profile: str,
    ) -> None:
        material_library_path.write_text("#usda 1.0\n", encoding="utf-8")
        authored.append((material_library_path, tuple(materials), material_profile))

    monkeypatch.setattr(
        generated_library_builder,
        "generate_texture_maps",
        fake_generate_texture_maps,
    )
    monkeypatch.setattr(
        generated_library_builder,
        "write_material_library_usd",
        fake_write_material_library_usd,
    )

    image_model = object()
    settings = TextureGenerationSettings(texture_size=4)
    package = build_generated_material_library(
        MaterialGenerationPlan(materials=(_blue_plastic_recipe(),)),
        tmp_path / "package",
        texture_settings=settings,
        image_model=image_model,
        prototype_materials_data={
            "library_path": "prototype.usda",
            "entries": [
                {
                    "name": "Blue Glossy Plastic",
                    "description": "saturated blue glossy plastic body shell",
                    "binding": "/World/Looks/Blue",
                }
            ],
        },
        prototype_min_score=0.0,
        material_profile="openpbr",
        include_generation_metadata=False,
    )

    assert calls == [("blue_glossy_plastic", settings, image_model)]
    assert authored[0][2] == "openpbr_materialx"
    assert package.material_library_path.exists()
    assert package.generation_plan_path is not None
    manifest = yaml.safe_load(
        package.materials_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["entries"] == [
        {
            "name": "Generated Blue Glossy Plastic",
            "description": _blue_plastic_recipe().description,
            "binding": "/World/Looks/Generated_Blue_Glossy_Plastic",
        }
    ]
    debug_plan = yaml.safe_load(
        package.generation_plan_path.read_text(encoding="utf-8")
    )
    assert debug_plan["materials"][0]["generated_textures"]["albedo"] == (
        "textures/blue_glossy_plastic/albedo.png"
    )
    assert package.materials[0].prototype_source["name"] == "Blue Glossy Plastic"


def test_build_generated_material_library_can_skip_debug_plan_and_load_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_generate_texture_maps(
        recipe: MaterialRecipe,
        output_dir: Path,
        *,
        settings: TextureGenerationSettings | None,
        image_model: object | None,
    ) -> TextureMapSet:
        output_dir.mkdir(parents=True)
        maps = TextureMapSet(
            albedo=output_dir / "albedo.png",
            normal=output_dir / "normal.png",
            orm=output_dir / "orm.png",
        )
        for path in (maps.albedo, maps.normal, maps.orm):
            path.write_bytes(b"png")
        return maps

    monkeypatch.setattr(
        generated_library_builder,
        "generate_texture_maps",
        fake_generate_texture_maps,
    )
    monkeypatch.setattr(
        generated_library_builder,
        "write_material_library_usd",
        lambda path, materials, *, material_profile: path.write_text(
            "#usda 1.0\n", encoding="utf-8"
        ),
    )
    prototype_manifest = tmp_path / "prototype_materials.yaml"
    prototype_manifest.write_text(
        yaml.safe_dump(
            {
                "library_path": "prototype.usda",
                "entries": [
                    {
                        "name": "Blue Glossy Plastic",
                        "description": "saturated blue glossy plastic",
                        "binding": "/World/Looks/Blue",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    package = build_generated_material_library(
        MaterialGenerationPlan(materials=(_blue_plastic_recipe(),)),
        tmp_path / "package_without_debug",
        prototype_materials_path=prototype_manifest,
        write_debug_plan=False,
        prototype_min_score=999.0,
    )

    assert package.generation_plan_path is None
    assert package.materials[0].prototype_source is None


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_builds_valid_generated_material_library_package(tmp_path: Path) -> None:
    plan = MaterialGenerationPlan(
        asset={"usd_path": "/tmp/input.usd", "asset_summary": "test asset"},
        materials=(_blue_plastic_recipe(), _brushed_aluminum_recipe()),
    )

    library = build_generated_material_library(
        plan,
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=16, seed=7),
    )

    assert library.material_library_path.exists()
    assert library.materials_manifest_path.exists()
    assert library.generation_plan_path is not None
    assert library.generation_plan_path.exists()

    manifest = yaml.safe_load(
        library.materials_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["library_path"] == "material_library.usda"
    assert [entry["name"] for entry in manifest["entries"]] == [
        "Generated Blue Glossy Plastic",
        "Generated Brushed Aluminum Fine Grain",
    ]

    generation_plan = yaml.safe_load(
        library.generation_plan_path.read_text(encoding="utf-8")
    )
    assert generation_plan["materials"][0]["generated_textures"]["albedo"] == (
        "textures/blue_glossy_plastic/albedo.png"
    )

    result = validate_generated_material_library(library.materials_manifest_path)
    assert result.ok, result.errors
    assert result.metadata["entry_count"] == 2
    assert result.metadata["texture_count"] == 6

    stage = Usd.Stage.Open(str(library.material_library_path))
    assert stage is not None
    material = UsdShade.Material(
        stage.GetPrimAtPath("/World/Looks/Generated_Blue_Glossy_Plastic")
    )
    assert material
    assert material.GetSurfaceOutput().HasConnectedSource()
    shader = UsdShade.Shader(
        stage.GetPrimAtPath("/World/Looks/Generated_Blue_Glossy_Plastic/PreviewSurface")
    )
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert list(shader.GetInput("diffuseColor").Get()) == pytest.approx(
        [0.001548, 0.019607, 0.692071], abs=1e-6
    )

    albedo_texture = stage.GetPrimAtPath(
        "/World/Looks/Generated_Blue_Glossy_Plastic/AlbedoTexture"
    )
    albedo_file = albedo_texture.GetAttribute("inputs:file")
    assert albedo_file.Get().path == "textures/blue_glossy_plastic/albedo.png"
    assert list(albedo_texture.GetAttribute("inputs:fallback").Get()) == pytest.approx(
        [0.02, 0.15, 0.85, 1.0]
    )

    normal_texture = stage.GetPrimAtPath(
        "/World/Looks/Generated_Blue_Glossy_Plastic/NormalTexture"
    )
    assert list(normal_texture.GetAttribute("inputs:fallback").Get()) == [
        0.5,
        0.5,
        1.0,
        1.0,
    ]
    assert list(normal_texture.GetAttribute("inputs:scale").Get()) == [
        2.0,
        2.0,
        2.0,
        1.0,
    ]
    assert list(normal_texture.GetAttribute("inputs:bias").Get()) == [
        -1.0,
        -1.0,
        -1.0,
        0.0,
    ]


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_generated_material_library_can_author_omnipbr_mdl_profile(
    tmp_path: Path,
) -> None:
    library = build_generated_material_library(
        MaterialGenerationPlan(materials=(_blue_plastic_recipe(),)),
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=8, seed=19),
        material_profile="omnipbr_mdl",
    )

    stage = Usd.Stage.Open(str(library.material_library_path))
    material_path = "/World/Looks/Generated_Blue_Glossy_Plastic"
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    assert material.GetSurfaceOutput("mdl").HasConnectedSource()
    assert material.GetSurfaceOutput().HasConnectedSource()

    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{material_path}/MDLShader"))
    shader_prim = shader.GetPrim()
    assert shader_prim.IsValid()
    assert shader_prim.GetAttribute("info:mdl:sourceAsset").Get().path == "OmniPBR.mdl"
    assert (
        shader_prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
        == "OmniPBR"
    )
    assert material.GetInput("DiffuseTexture").Get().path == (
        "textures/blue_glossy_plastic/albedo.png"
    )
    assert material.GetInput("NormalTexture").Get().path == (
        "textures/blue_glossy_plastic/normal.png"
    )
    assert material.GetInput("ORMTexture").Get().path == (
        "textures/blue_glossy_plastic/orm.png"
    )


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_generated_material_library_openpbr_synthesis_requires_usdex_helper(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="usdex.core.definePbrMaterial"):
        build_generated_material_library(
            MaterialGenerationPlan(materials=(_blue_plastic_recipe(),)),
            tmp_path / "generated_material_library",
            texture_settings=TextureGenerationSettings(texture_size=8, seed=29),
            material_profile="openpbr_materialx",
        )


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_generated_material_can_derive_openpbr_from_default_library(
    tmp_path: Path,
) -> None:
    recipe = _off_white_satin_plastic_recipe()
    prototypes = load_material_prototypes_from_manifest(
        _default_materials_manifest_path()
    )
    selected = select_material_prototype(recipe, prototypes)
    assert selected is not None
    assert selected[0].name == "Car Paint Pure White"

    library = build_generated_material_library(
        MaterialGenerationPlan(materials=(recipe,)),
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=8, seed=5),
        prototype_materials_path=_default_materials_manifest_path(),
    )

    manifest = yaml.safe_load(
        library.materials_manifest_path.read_text(encoding="utf-8")
    )
    entry = manifest["entries"][0]
    assert entry["prototype_source"]["name"] == "Car Paint Pure White"

    result = validate_generated_material_library(library.materials_manifest_path)
    assert result.ok, result.errors

    stage = Usd.Stage.Open(str(library.material_library_path))
    material_prim = stage.GetPrimAtPath("/World/Looks/Off_White_Satin_Plastic")
    assert material_prim.IsValid()
    assert stage.GetPrimAtPath(
        "/World/Looks/Off_White_Satin_Plastic/open_pbr_surface_surfaceshader"
    ).IsValid()
    assert not stage.GetPrimAtPath(
        "/World/Looks/Off_White_Satin_Plastic/PreviewSurface"
    ).IsValid()

    material = UsdShade.Material(material_prim)
    assert list(material.GetInput("base_color").Get()) == pytest.approx(
        [0.848088, 0.827571, 0.787412]
    )
    albedo_input = material.GetInput("base_color_texture_file")
    assert albedo_input.Get().path == ("textures/off_white_satin_plastic/albedo.png")
    assert albedo_input.GetAttr().GetColorSpace() == "auto"
    assert material.GetInput("base_metalness").Get() == pytest.approx(0.0)
    assert material.GetInput("specular_roughness").Get() == pytest.approx(0.4)
    assert material.GetInput("coat_weight").Get() == pytest.approx(1.0)
    assert material.GetInput("coat_roughness").Get() == pytest.approx(0.02)
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert material.GetSurfaceOutput("mtlx").HasConnectedSource()
    assert stage.GetPrimAtPath(
        "/World/Looks/Off_White_Satin_Plastic/OVRTXPreviewSurface"
    ).IsValid()


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_prototype_copy_remaps_relative_assets_and_authors_generated_albedo(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source_library"
    source_dir.mkdir()
    texture_dir = source_dir / "textures"
    texture_dir.mkdir()
    Image.new("RGB", (4, 4), (128, 128, 255)).save(texture_dir / "prototype_normal.png")

    prototype_usd = source_dir / "prototype_library.usda"
    stage = Usd.Stage.CreateNew(str(prototype_usd))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Prototype_Blue_Plastic")
    material.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.02, 0.15, 0.85)
    )
    material.CreateInput("base_metalness", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    material.CreateInput("normal_texture_file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/prototype_normal.png")
    )
    stage.GetRootLayer().Save()

    manifest_path = source_dir / "materials.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "library_path": "prototype_library.usda",
                "entries": [
                    {
                        "name": "Generated Blue Glossy Plastic",
                        "binding": "/World/Looks/Prototype_Blue_Plastic",
                        "description": "saturated blue glossy molded plastic",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    library = build_generated_material_library(
        MaterialGenerationPlan(materials=(_blue_plastic_recipe(),)),
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=8, seed=13),
        prototype_materials_path=manifest_path,
        prototype_min_score=0.0,
    )

    result = validate_generated_material_library(library.materials_manifest_path)
    assert result.ok, result.errors

    generated_stage = Usd.Stage.Open(str(library.material_library_path))
    generated_material = UsdShade.Material(
        generated_stage.GetPrimAtPath("/World/Looks/Generated_Blue_Glossy_Plastic")
    )
    assert generated_material

    normal_input = generated_material.GetInput("normal_texture_file")
    assert normal_input.Get().path == (
        "prototype_assets/blue_glossy_plastic/textures/prototype_normal.png"
    )

    albedo_input = generated_material.GetInput("base_color_texture_file")
    assert albedo_input.Get().path == "textures/blue_glossy_plastic/albedo.png"
    assert albedo_input.GetAttr().GetColorSpace() == "auto"


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_non_optical_generated_material_resets_prototype_transmission(
    tmp_path: Path,
) -> None:
    recipe = _matte_black_rubber_recipe()
    prototypes = load_material_prototypes_from_manifest(
        _default_materials_manifest_path()
    )
    selected = select_material_prototype(recipe, prototypes)
    assert selected is not None
    assert selected[0].name == "Rubber Black Matte"

    library = build_generated_material_library(
        MaterialGenerationPlan(materials=(recipe,)),
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=8, seed=11),
        prototype_materials_path=_default_materials_manifest_path(),
    )

    stage = Usd.Stage.Open(str(library.material_library_path))
    material = UsdShade.Material(stage.GetPrimAtPath(recipe.binding))
    assert material
    expected_base_color = _srgb_to_linear_tuple(recipe.base_color_hint)

    assert material.GetInput("transmission_weight").Get() == pytest.approx(0.0)
    assert material.GetInput("geometry_opacity").Get() == pytest.approx(1.0)
    assert material.GetInput("geometry_thin_walled").Get() is False
    assert list(material.GetInput("transmission_color").Get()) == pytest.approx(
        expected_base_color,
    )
    assert material.GetInput("transmission_weight_texture_file").Get().path == ""
    assert material.GetInput("transmission_color_texture_file").Get().path == ""
    assert material.GetInput("geometry_opacity_texture_file").Get().path == ""

    shader = UsdShade.Shader(
        stage.GetPrimAtPath(f"{recipe.binding}/OVRTXPreviewSurface")
    )
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert list(shader.GetInput("diffuseColor").Get()) == pytest.approx(
        expected_base_color,
    )
    assert shader.GetInput("opacity").Get() == pytest.approx(1.0)
    assert _connected_surface_shader_id(material) == "UsdPreviewSurface"
    assert material.GetSurfaceOutput("mtlx").HasConnectedSource()


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_translucent_generated_material_prefers_glass_prototype(
    tmp_path: Path,
) -> None:
    recipe = _frosted_glass_recipe()
    prototypes = load_material_prototypes_from_manifest(
        _default_materials_manifest_path()
    )
    selected = select_material_prototype(recipe, prototypes)
    assert selected is not None
    assert selected[0].name == "Glass Frosted"

    library = build_generated_material_library(
        MaterialGenerationPlan(materials=(recipe,)),
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=8, seed=23),
        prototype_materials_path=_default_materials_manifest_path(),
    )

    manifest = yaml.safe_load(
        library.materials_manifest_path.read_text(encoding="utf-8")
    )
    entry = manifest["entries"][0]
    assert entry["prototype_source"]["name"] == "Glass Frosted"

    stage = Usd.Stage.Open(str(library.material_library_path))
    material = UsdShade.Material(stage.GetPrimAtPath(recipe.binding))
    assert material
    assert material.GetInput("base_metalness").Get() == pytest.approx(0.0)
    assert material.GetInput("transmission_weight").Get() == pytest.approx(1.0)
    assert material.GetInput("geometry_opacity").Get() == pytest.approx(1.0)
    assert material.GetInput("specular_ior").Get() == pytest.approx(1.48)
    assert material.GetInput("specular_roughness").Get() == pytest.approx(0.5)
    assert material.GetInput("coat_weight").Get() == pytest.approx(0.0)
    albedo_input = material.GetInput("base_color_texture_file")
    assert albedo_input.Get().path == ""


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_build_material_library_falls_back_when_image_model_returns_no_image(
    tmp_path: Path,
) -> None:
    class NoImageModel:
        def generate(self, *args, **kwargs):
            raise ValueError("No image generated in response")

    plan = MaterialGenerationPlan(materials=(_blue_plastic_recipe(),))

    library = build_generated_material_library(
        plan,
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=8, seed=3),
        image_model=NoImageModel(),
    )

    assert library.material_library_path.exists()
    assert library.materials_manifest_path.exists()
    assert (library.package_dir / "textures/blue_glossy_plastic/albedo.png").exists()
    assert library.generation_plan_path is not None
    generation_plan = yaml.safe_load(
        library.generation_plan_path.read_text(encoding="utf-8")
    )
    assert generation_plan["materials"][0]["generated_textures"]["albedo"] == (
        "textures/blue_glossy_plastic/albedo.png"
    )


def test_generated_albedo_is_color_corrected_to_base_color_hint(
    tmp_path: Path,
) -> None:
    class WrongColorImageModel:
        def generate(self, *args, **kwargs):
            image = Image.new("RGB", (8, 8), (50, 180, 40))
            for index in range(8):
                image.putpixel((index, index), (70, 200, 60))
            return image

    recipe = _blue_plastic_recipe()
    maps = generate_texture_maps(
        recipe,
        tmp_path / "textures",
        settings=TextureGenerationSettings(
            texture_size=8,
            color_correct_albedo=True,
            albedo_color_correction_strength=1.0,
        ),
        image_model=WrongColorImageModel(),
    )

    assert _mean_rgb(maps.albedo) == pytest.approx(
        recipe.base_color_hint,
        abs=1 / 255,
    )


def test_generated_albedo_color_correction_can_be_disabled(tmp_path: Path) -> None:
    class WrongColorImageModel:
        def generate(self, *args, **kwargs):
            return Image.new("RGB", (8, 8), (50, 180, 40))

    recipe = _blue_plastic_recipe()
    maps = generate_texture_maps(
        recipe,
        tmp_path / "textures",
        settings=TextureGenerationSettings(
            texture_size=8,
            color_correct_albedo=False,
        ),
        image_model=WrongColorImageModel(),
    )

    assert _mean_rgb(maps.albedo) == pytest.approx(
        (50 / 255, 180 / 255, 40 / 255),
        abs=1 / 255,
    )


def test_generate_material_library_task_parses_albedo_color_correction_config() -> None:
    settings = GenerateMaterialLibraryTask()._texture_settings_from_context(
        {
            "texture_generation": {
                "texture_size": 16,
                "color_correct_albedo": False,
                "albedo_color_correction_strength": 0.25,
            }
        }
    )

    assert settings.texture_size == 16
    assert settings.color_correct_albedo is False
    assert settings.albedo_color_correction_strength == pytest.approx(0.25)


def test_synthesized_albedo_is_deterministic_and_nonblank(tmp_path: Path) -> None:
    settings = TextureGenerationSettings(texture_size=16, seed=17)

    first = generate_texture_maps(
        _blue_plastic_recipe(),
        tmp_path / "first",
        settings=settings,
    )
    second = generate_texture_maps(
        _blue_plastic_recipe(),
        tmp_path / "second",
        settings=settings,
    )

    assert first.albedo.read_bytes() == second.albedo.read_bytes()
    image = Image.open(first.albedo).convert("RGB")
    assert image.getbbox() is not None
    assert any(low != high for low, high in image.getextrema())


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_generate_material_library_task_uses_default_prototype_data(
    tmp_path: Path,
) -> None:
    prototype_data = yaml.safe_load(
        _default_materials_manifest_path().read_text(encoding="utf-8")
    )
    prototype_data["library_path"] = str(
        (
            _default_materials_manifest_path().parent / prototype_data["library_path"]
        ).resolve()
    )

    result = GenerateMaterialLibraryTask().run(
        {
            "material_generation_plan": MaterialGenerationPlan(
                materials=(_off_white_satin_plastic_recipe(),)
            ).to_dict(),
            "output_dir": str(tmp_path / "generated_material_library"),
            "texture_generation": {"texture_size": 8, "seed": 17},
            "prototype_materials_data": prototype_data,
            "material_authoring": {"use_default_prototypes": True},
        }
    )

    entry = result["generated_materials_data"]["entries"][0]
    assert entry["prototype_source"]["name"] == "Car Paint Pure White"


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_generate_material_library_task_loads_plan_yaml(tmp_path: Path) -> None:
    plan_path = tmp_path / "material_generation_plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            MaterialGenerationPlan(materials=(_blue_plastic_recipe(),)).to_dict()
        ),
        encoding="utf-8",
    )

    result = GenerateMaterialLibraryTask().run(
        {
            "material_generation_plan_path": str(plan_path),
            "output_dir": str(tmp_path / "generated_material_library"),
            "texture_generation": {"texture_size": 8, "seed": 11},
        }
    )

    assert Path(result["generated_material_library_path"]).exists()
    assert Path(result["generated_materials_yaml_path"]).exists()
    assert result["generated_materials_data"]["entries"][0]["name"] == (
        "Generated Blue Glossy Plastic"
    )
    assert result["generation_validation"]["ok"] is True


@pytest.mark.skipif(
    os.getenv("RUN_USDEX_TESTS") != "1",
    reason="usdex.core aborts the Python process in this test environment",
)
def test_generated_material_library_consumes_through_apply_pipeline(
    tmp_path: Path,
) -> None:
    plan = MaterialGenerationPlan(materials=(_blue_plastic_recipe(),))
    library = build_generated_material_library(
        plan,
        tmp_path / "generated_material_library",
        texture_settings=TextureGenerationSettings(texture_size=8),
    )

    input_usd = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_usd))
    root = UsdGeom.Xform.Define(stage, "/RootNode")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/RootNode/Geometry")
    UsdGeom.Mesh.Define(stage, "/RootNode/Geometry/OuterShell")
    stage.GetRootLayer().Save()

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "/RootNode/Geometry/OuterShell",
                "materials": {"material": "Generated Blue Glossy Plastic"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    context = {
        "unique_materials": ["Generated Blue Glossy Plastic"],
        "materials_mapping": _material_mapping_from_manifest(
            library.materials_manifest_path
        ),
    }
    context = MaterialRetrievalTask().run(context)
    context = ResolveMaterialFilesTask().run(context)

    output_usd = tmp_path / "output.usda"
    apply_context = {
        **context,
        "input_usd_path": str(input_usd),
        "output_usd_path": str(output_usd),
        "predictions_path": str(predictions_path),
        "layer_only": False,
        "flatten_output": False,
    }
    result = ApplyMaterialsToUSDTask().run(apply_context)

    assert result["materials_applied"] == {
        "Generated Blue Glossy Plastic": "/RootNode/Looks/Generated_Blue_Glossy_Plastic"
    }
    output_stage = Usd.Stage.Open(str(output_usd))
    assert output_stage is not None
    assert output_stage.GetPrimAtPath(
        "/RootNode/Looks/Generated_Blue_Glossy_Plastic"
    ).IsValid()

    bound_material, _ = UsdShade.MaterialBindingAPI(
        output_stage.GetPrimAtPath("/RootNode/Geometry/OuterShell")
    ).ComputeBoundMaterial()
    assert bound_material
    assert str(bound_material.GetPath()) == (
        "/RootNode/Looks/Generated_Blue_Glossy_Plastic"
    )
