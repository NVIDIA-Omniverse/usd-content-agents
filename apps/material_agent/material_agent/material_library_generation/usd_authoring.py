# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD authoring for generated material libraries."""

from __future__ import annotations

import importlib
import os
import re
import shutil
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from world_understanding.utils.usd.asset_paths import (
    is_absolute_asset_path,
    is_relative_to,
    is_uri_asset_path,
    resolve_relative_asset_path_under_base,
)

from material_agent.material_library_generation.schema import (
    DEFAULT_LIBRARY_ROOT,
    GeneratedMaterial,
)
from material_agent.material_profiles import MaterialProfile, normalize_material_profile

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OPTICAL_TOKENS = {
    "acrylic",
    "clear",
    "cloudy",
    "frosted",
    "glass",
    "milky",
    "transparent",
    "translucent",
}
_OPENPBR_MATERIALX_SHADER_ID = "ND_open_pbr_surface_surfaceshader"
_OPENPBR_IMAGE_SHADER_IDS = frozenset(
    {
        "ND_image_color3",
        "ND_image_vector3",
        "ND_tiledimage_color3",
        "ND_tiledimage_vector3",
    }
)
_OPENPBR_USDEX_FUNCTIONS = (
    "definePbrMaterial",
    "addDiffuseTextureToPbrMaterial",
    "addNormalTextureToPbrMaterial",
    "addOrmTextureToPbrMaterial",
)
_TEXTURE_SUFFIXES = frozenset({".bmp", ".exr", ".jpeg", ".jpg", ".png", ".tga"})


@dataclass(frozen=True)
class OpenPbrMaterialXAuthoringCapability:
    """Installed USD-Exchange support required by the textured OpenPBR path."""

    available: bool
    installed_version: str | None
    missing_symbols: tuple[str, ...]
    import_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "installed_version": self.installed_version,
            "missing_symbols": list(self.missing_symbols),
            "import_error": self.import_error,
        }


class MaterialAuthoringError(ValueError):
    """Structured authoring failure that prevents package registration."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": dict(self.details),
        }


class MaterialAuthoringPrerequisiteError(MaterialAuthoringError):
    """The installed authoring runtime cannot satisfy an explicit profile."""


class MaterialAuthoringContractError(MaterialAuthoringError):
    """The authored USD does not satisfy the requested material contract."""


def _relative_asset_path(asset_path: Path, usd_path: Path) -> str:
    relative = os.path.relpath(asset_path.resolve(), usd_path.resolve().parent)
    return relative.replace("\\", "/")


def _recipe_text(generated: GeneratedMaterial) -> str:
    recipe = generated.recipe
    parts = [
        recipe.name,
        recipe.description,
        recipe.appearance_prompt,
        recipe.color or "",
        recipe.material or "",
        recipe.finish or "",
    ]
    parts.extend(part.semantic_label for part in recipe.intended_parts)
    parts.extend(part.evidence for part in recipe.intended_parts)
    return " ".join(parts).lower()


def _recipe_tokens(generated: GeneratedMaterial) -> set[str]:
    return set(_TOKEN_RE.findall(_recipe_text(generated)))


def _srgb_channel_to_linear(channel: float) -> float:
    channel = max(0.0, min(1.0, float(channel)))
    if channel <= 0.04045:
        return channel / 12.92
    return float(((channel + 0.055) / 1.055) ** 2.4)


def _srgb_color_to_linear(
    color: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        _srgb_channel_to_linear(color[0]),
        _srgb_channel_to_linear(color[1]),
        _srgb_channel_to_linear(color[2]),
    )


def _is_matte(generated: GeneratedMaterial) -> bool:
    text = _recipe_text(generated)
    return any(token in text for token in ("matte", "flat", "dull", "rough"))


def _is_metal(generated: GeneratedMaterial) -> bool:
    recipe = generated.recipe
    if recipe.pbr_hints.metallic >= 0.5:
        return True
    text = _recipe_text(generated)
    return any(
        token in text
        for token in (
            "aluminum",
            "brass",
            "bronze",
            "copper",
            "iron",
            "metal",
            "metallic",
            "silver",
            "steel",
        )
    )


def _is_optical(generated: GeneratedMaterial) -> bool:
    recipe = generated.recipe
    if recipe.pbr_hints.transmission > 0.0 or recipe.pbr_hints.opacity < 1.0:
        return True
    return bool(_recipe_tokens(generated) & _OPTICAL_TOKENS)


def _optical_transmission(generated: GeneratedMaterial) -> float:
    transmission = generated.recipe.pbr_hints.transmission
    if transmission > 0.0:
        return float(transmission)
    return 1.0


def _optical_opacity(generated: GeneratedMaterial) -> float:
    return float(generated.recipe.pbr_hints.opacity)


def _optical_roughness(generated: GeneratedMaterial, roughness: float) -> float:
    tokens = _recipe_tokens(generated)
    if tokens & {"cloudy", "frosted", "milky", "translucent"}:
        return max(roughness, 0.5)
    return roughness


def _ensure_parent_specs(layer: Any, path: Any, Sdf: Any) -> None:
    parent = path.GetParentPath()
    parent_paths = []
    while parent != Sdf.Path.absoluteRootPath:
        parent_paths.append(parent)
        parent = parent.GetParentPath()
    for parent_path in reversed(parent_paths):
        if not layer.GetPrimAtPath(parent_path):
            Sdf.CreatePrimInLayer(layer, parent_path)


def _copy_prototype_asset_path(
    path_str: str,
    *,
    source_dir: Path,
    package_dir: Path,
    asset_subdir: Path,
) -> str:
    if not path_str or is_uri_asset_path(path_str):
        return ""

    try:
        if is_absolute_asset_path(path_str):
            source_path = Path(path_str).resolve()
            if not is_relative_to(source_path, source_dir):
                return ""
        else:
            source_path = resolve_relative_asset_path_under_base(path_str, source_dir)
    except ValueError:
        return ""

    if not source_path.exists() or not source_path.is_file():
        return ""

    try:
        relative_source = source_path.relative_to(source_dir)
    except ValueError:
        return ""

    target_path = package_dir / asset_subdir / relative_source
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return _relative_asset_path(target_path, package_dir / "material_library.usda")


def _copy_prototype_assets_into_package(
    layer: Any,
    prim_path: Any,
    *,
    source_dir: Path,
    package_dir: Path,
    asset_subdir: Path,
    Sdf: Any,
) -> None:
    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return

    for attr_name in list(prim_spec.attributes.keys()):
        attr_spec = prim_spec.attributes[attr_name]
        value = attr_spec.default
        if isinstance(value, Sdf.AssetPath):
            copied_path = _copy_prototype_asset_path(
                value.path,
                source_dir=source_dir,
                package_dir=package_dir,
                asset_subdir=asset_subdir,
            )
            if copied_path != value.path:
                attr_spec.default = Sdf.AssetPath(copied_path)
        elif isinstance(value, Sdf.AssetPathArray):
            copied_paths = [
                Sdf.AssetPath(
                    _copy_prototype_asset_path(
                        asset_path.path,
                        source_dir=source_dir,
                        package_dir=package_dir,
                        asset_subdir=asset_subdir,
                    )
                )
                for asset_path in value
            ]
            copied_array = Sdf.AssetPathArray(copied_paths)
            if copied_array != value:
                attr_spec.default = copied_array

    for child_spec in prim_spec.nameChildren:
        _copy_prototype_assets_into_package(
            layer,
            prim_path.AppendChild(child_spec.name),
            source_dir=source_dir,
            package_dir=package_dir,
            asset_subdir=asset_subdir,
            Sdf=Sdf,
        )


def _set_existing_material_input(material: Any, name: str, value: Any) -> None:
    material_input = material.GetInput(name)
    if material_input:
        material_input.Set(value)


def _set_existing_asset_material_input(
    material: Any,
    name: str,
    value: Any,
    *,
    color_space: str | None = None,
    create_missing: bool = False,
    sdf: Any | None = None,
) -> None:
    material_input = material.GetInput(name)
    if not material_input:
        if not create_missing or sdf is None:
            return
        material_input = material.CreateInput(name, sdf.ValueTypeNames.Asset)
    material_input.Set(value)
    if color_space is not None:
        material_input.GetAttr().SetColorSpace(color_space)


def _clear_existing_asset_material_input(material: Any, name: str, Sdf: Any) -> None:
    _set_existing_asset_material_input(material, name, Sdf.AssetPath(""))


def _set_shader_input_if_present(prim: Any, name: str, value: Any) -> None:
    from pxr import UsdShade

    for child in prim.GetChildren():
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        shader_input = shader.GetInput(name)
        if shader_input:
            shader_input.Set(value)


def _set_albedo_asset_input_if_present(prim: Any, asset_path: str, Sdf: Any) -> None:
    from pxr import UsdShade

    for child in prim.GetChildren():
        if child.GetName() not in {"AlbedoTexture", "tiledimage_base_color"}:
            continue
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        shader_input = shader.GetInput("file")
        if shader_input and shader_input.GetTypeName() == Sdf.ValueTypeNames.Asset:
            shader_input.Set(Sdf.AssetPath(asset_path))
            shader_input.GetAttr().SetColorSpace("auto")
        source_color_space = shader.GetInput("sourceColorSpace")
        if source_color_space:
            source_color_space.Set("auto")


def _adapt_openpbr_material(
    stage: Any,
    material_path: str,
    generated: GeneratedMaterial,
    library_path: Path,
    Sdf: Any,
) -> None:
    from pxr import Gf, UsdShade

    prim = stage.GetPrimAtPath(material_path)
    material = UsdShade.Material(prim)
    recipe = generated.recipe
    optical = _is_optical(generated)
    base_color = Gf.Vec3f(*_srgb_color_to_linear(recipe.base_color_hint))
    roughness = _optical_roughness(generated, float(recipe.pbr_hints.roughness))
    metallic = float(recipe.pbr_hints.metallic if _is_metal(generated) else 0.0)
    albedo_asset = _relative_asset_path(generated.textures.albedo, library_path)

    _set_existing_material_input(material, "base_color", base_color)
    _set_existing_material_input(material, "base_metalness", metallic)
    _set_existing_material_input(material, "specular_roughness", roughness)
    _set_existing_material_input(material, "roughness", roughness)
    if not optical:
        _set_existing_asset_material_input(
            material,
            "base_color_texture_file",
            Sdf.AssetPath(albedo_asset),
            color_space="auto",
            create_missing=True,
            sdf=Sdf,
        )
        _set_existing_material_input(material, "transmission_weight", 0.0)
        _set_existing_material_input(material, "transmission_color", base_color)
        _set_existing_material_input(material, "geometry_opacity", 1.0)
        _set_existing_material_input(material, "geometry_thin_walled", False)
        _clear_existing_asset_material_input(
            material,
            "transmission_weight_texture_file",
            Sdf,
        )
        _clear_existing_asset_material_input(
            material,
            "transmission_color_texture_file",
            Sdf,
        )
        _clear_existing_asset_material_input(
            material,
            "geometry_opacity_texture_file",
            Sdf,
        )

    if optical:
        _set_existing_material_input(
            material,
            "transmission_weight",
            _optical_transmission(generated),
        )
        _set_existing_material_input(
            material,
            "geometry_opacity",
            _optical_opacity(generated),
        )
        _set_existing_material_input(
            material,
            "geometry_thin_walled",
            bool(recipe.pbr_hints.thin_walled),
        )
        _set_existing_material_input(
            material,
            "specular_ior",
            float(recipe.pbr_hints.ior),
        )
        _set_existing_material_input(material, "specular_weight", 1.0)
        _set_existing_material_input(material, "coat_weight", 0.0)

    if _is_matte(generated):
        _set_existing_material_input(material, "coat_weight", 0.0)
        _set_existing_material_input(material, "coat_roughness", roughness)

    _set_shader_input_if_present(prim, "diffuseColor", base_color)
    _set_shader_input_if_present(prim, "metallic", metallic)
    _set_shader_input_if_present(prim, "roughness", roughness)
    if not optical:
        _set_albedo_asset_input_if_present(prim, albedo_asset, Sdf)
        _set_shader_input_if_present(prim, "transmission_weight", 0.0)
        _set_shader_input_if_present(prim, "transmission_color", base_color)
        _set_shader_input_if_present(prim, "geometry_opacity", 1.0)


def _try_author_from_prototype(
    stage: Any,
    library_path: Path,
    generated: GeneratedMaterial,
) -> bool:
    from pxr import Sdf

    prototype = generated.prototype_source or {}
    source_library_path = prototype.get("library_path")
    source_binding = prototype.get("binding")
    if not source_library_path or not source_binding:
        return False

    source_layer = Sdf.Layer.FindOrOpen(str(source_library_path))
    if not source_layer:
        return False
    if not source_layer.GetPrimAtPath(str(source_binding)):
        return False

    layer = stage.GetRootLayer()
    target_path = Sdf.Path(generated.binding)
    _ensure_parent_specs(layer, target_path, Sdf)
    if not Sdf.CopySpec(
        source_layer,
        Sdf.Path(str(source_binding)),
        layer,
        target_path,
    ):
        return False

    from material_agent.tasks.apply_materials_to_usd import (
        clear_color_space_on_empty_asset_inputs,
    )

    _copy_prototype_assets_into_package(
        layer,
        target_path,
        source_dir=Path(source_library_path).resolve().parent,
        package_dir=library_path.resolve().parent,
        asset_subdir=Path("prototype_assets") / generated.recipe.material_id,
        Sdf=Sdf,
    )
    clear_color_space_on_empty_asset_inputs(layer, target_path)
    _adapt_openpbr_material(stage, generated.binding, generated, library_path, Sdf)
    return True


def _connect_texture_st(texture_shader: Any, st_reader: Any, Sdf: Any) -> None:
    texture_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(),
        "result",
    )


def _define_texture_shader(
    stage: Any,
    shader_path: str,
    texture_path: str,
    colorspace: str,
    fallback: tuple[float, float, float, float] | None = None,
) -> Any:
    from pxr import Gf, Sdf, UsdShade

    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(texture_path)
    )
    shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
    if fallback is not None:
        shader.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(*fallback)
        )
    shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
    shader.CreateOutput("g", Sdf.ValueTypeNames.Float)
    shader.CreateOutput("b", Sdf.ValueTypeNames.Float)
    return shader


def _installed_usd_exchange_version() -> str | None:
    try:
        return metadata.version("usd-exchange")
    except metadata.PackageNotFoundError:
        return None


def probe_openpbr_materialx_authoring() -> OpenPbrMaterialXAuthoringCapability:
    """Probe the complete USD-Exchange API needed for textured OpenPBR."""

    installed_version = _installed_usd_exchange_version()
    try:
        core = importlib.import_module("usdex.core")
    except Exception as exc:
        return OpenPbrMaterialXAuthoringCapability(
            available=False,
            installed_version=installed_version,
            missing_symbols=tuple(
                f"usdex.core.{name}" for name in _OPENPBR_USDEX_FUNCTIONS
            ),
            import_error=f"{type(exc).__name__}: {exc}",
        )

    missing_symbols = tuple(
        f"usdex.core.{name}"
        for name in _OPENPBR_USDEX_FUNCTIONS
        if not callable(getattr(core, name, None))
    )
    return OpenPbrMaterialXAuthoringCapability(
        available=not missing_symbols,
        installed_version=installed_version,
        missing_symbols=missing_symbols,
    )


def can_author_openpbr_materialx_with_usdex() -> bool:
    """Return whether USD-Exchange can author the complete textured graph."""

    return probe_openpbr_materialx_authoring().available


def require_material_authoring_prerequisites(
    material_profile: str | MaterialProfile,
) -> None:
    """Fail before generation when an explicit authoring profile is unavailable."""

    normalized = normalize_material_profile(material_profile)
    if normalized != "openpbr_materialx":
        return

    capability = probe_openpbr_materialx_authoring()
    if capability.available:
        return

    version = capability.installed_version or "not installed"
    missing = ", ".join(capability.missing_symbols) or "unknown capability"
    raise MaterialAuthoringPrerequisiteError(
        "OPENPBR_MATERIALX_AUTHORING_UNAVAILABLE",
        "material_profile='openpbr_materialx' requires the public USD-Exchange "
        "core OpenPBR definition and texture attachment APIs tracked by issue "
        f"#371. Installed usd-exchange version: {version}. Missing: {missing}.",
        details={
            "requested_profile": normalized,
            "dependency_issue": 371,
            "usd_exchange": capability.to_dict(),
        },
    )


def _connect_preview_material_surface(
    material: Any, Sdf: Any, Usd: Any, UsdShade: Any
) -> None:
    """Ensure a generated PreviewSurface material has a universal surface output."""
    if material.GetSurfaceOutput().HasConnectedSource():
        return

    material_prim = material.GetPrim()
    for prim in Usd.PrimRange(material_prim):
        if prim == material_prim or not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        shader_id_attr = shader.GetIdAttr()
        if shader_id_attr and shader_id_attr.Get() == "UsdPreviewSurface":
            shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader_output)
            return

    raise ValueError(
        "Preview material authoring did not create a UsdPreviewSurface "
        f"shader under {material_prim.GetPath()}"
    )


def _define_preview_material_from_recipe(
    stage: Any,
    library_path: Path,
    generated: GeneratedMaterial,
    material_path: str,
    roughness: float,
    Sdf: Any,
) -> Any:
    from pxr import Gf, UsdShade

    recipe = generated.recipe
    optical = _is_optical(generated)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*_srgb_color_to_linear(recipe.base_color_hint))
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
        min(recipe.pbr_hints.opacity, 0.65) if optical else 1.0
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        float(recipe.pbr_hints.metallic)
    )
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)
    return material


def _define_omnipbr_mdl_material_from_recipe(
    stage: Any,
    library_path: Path,
    generated: GeneratedMaterial,
    material_path: str,
    roughness: float,
    Sdf: Any,
) -> Any:
    import usdex.rtx
    from pxr import Gf

    recipe = generated.recipe
    material = usdex.rtx.definePbrMaterial(
        stage,
        Sdf.Path(material_path),
        Gf.Vec3f(*_srgb_color_to_linear(recipe.base_color_hint)),
        float(recipe.pbr_hints.opacity),
        roughness,
        float(recipe.pbr_hints.metallic),
    )
    usdex.rtx.addDiffuseTextureToPbrMaterial(
        material,
        Sdf.AssetPath(_relative_asset_path(generated.textures.albedo, library_path)),
    )
    usdex.rtx.addNormalTextureToPbrMaterial(
        material,
        Sdf.AssetPath(_relative_asset_path(generated.textures.normal, library_path)),
    )
    usdex.rtx.addOrmTextureToPbrMaterial(
        material,
        Sdf.AssetPath(_relative_asset_path(generated.textures.orm, library_path)),
    )
    return material


def _define_openpbr_materialx_from_recipe(
    stage: Any,
    library_path: Path,
    generated: GeneratedMaterial,
    material_path: str,
    roughness: float,
    Sdf: Any,
) -> Any:
    from pxr import Gf

    core = importlib.import_module("usdex.core")

    recipe = generated.recipe
    material = core.definePbrMaterial(
        stage,
        Sdf.Path(material_path),
        Gf.Vec3f(*_srgb_color_to_linear(recipe.base_color_hint)),
        float(recipe.pbr_hints.opacity),
        roughness,
        float(recipe.pbr_hints.metallic),
    )
    if not material or not material.GetPrim().IsValid():
        raise MaterialAuthoringContractError(
            "OPENPBR_MATERIALX_DEFINITION_FAILED",
            "usdex.core.definePbrMaterial did not return a valid material.",
            details={
                "requested_profile": "openpbr_materialx",
                "material_path": material_path,
            },
        )

    core.addDiffuseTextureToPbrMaterial(
        material,
        Sdf.AssetPath(_relative_asset_path(generated.textures.albedo, library_path)),
    )
    core.addNormalTextureToPbrMaterial(
        material,
        Sdf.AssetPath(_relative_asset_path(generated.textures.normal, library_path)),
    )
    core.addOrmTextureToPbrMaterial(
        material,
        Sdf.AssetPath(_relative_asset_path(generated.textures.orm, library_path)),
    )
    return material


def _contract_error(
    code: str,
    message: str,
    *,
    material_path: str,
    details: dict[str, Any] | None = None,
) -> MaterialAuthoringContractError:
    error_details = {
        "material_path": material_path,
        **(details or {}),
    }
    return MaterialAuthoringContractError(code, message, details=error_details)


def _connected_source(port: Any, *, label: str, material_path: str) -> Any:
    if not port:
        raise _contract_error(
            "MATERIAL_CONNECTION_MISSING",
            f"Required material port is missing: {label}.",
            material_path=material_path,
            details={"port": label},
        )
    try:
        sources, invalid_sources = port.GetConnectedSources()
    except Exception as exc:
        raise _contract_error(
            "MATERIAL_CONNECTION_INVALID",
            f"Could not inspect material connection {label}: {exc}",
            material_path=material_path,
            details={"port": label},
        ) from exc
    if len(sources) != 1 or invalid_sources:
        raise _contract_error(
            "MATERIAL_CONNECTION_INVALID",
            f"Material connection {label} must have exactly one valid source.",
            material_path=material_path,
            details={
                "port": label,
                "source_count": len(sources),
                "invalid_source_paths": [str(path) for path in invalid_sources],
            },
        )
    return sources[0]


def _connected_shader(
    port: Any,
    *,
    label: str,
    material_path: str,
) -> tuple[Any, str]:
    from pxr import UsdShade

    source = _connected_source(port, label=label, material_path=material_path)
    shader = UsdShade.Shader(source.source.GetPrim())
    if not shader or not shader.GetPrim().IsValid():
        raise _contract_error(
            "MATERIAL_SHADER_SOURCE_INVALID",
            f"Material connection {label} does not resolve to a shader.",
            material_path=material_path,
            details={"port": label},
        )
    return shader, str(source.sourceName)


def _require_source_output(
    output_name: str,
    *,
    label: str,
    material_path: str,
    expected: str = "out",
) -> None:
    if output_name != expected:
        raise _contract_error(
            "MATERIAL_SOURCE_OUTPUT_INVALID",
            f"Material connection {label} must use outputs:{expected}.",
            material_path=material_path,
            details={
                "port": label,
                "expected_output": expected,
                "actual_output": output_name,
            },
        )


def _shader_id(shader: Any) -> str:
    shader_id = shader.GetIdAttr()
    return str(shader_id.Get()) if shader_id else ""


def _input_color_space(shader_input: Any) -> str:
    color_space = shader_input.GetAttr().GetColorSpace()
    return str(color_space) if color_space else ""


def _resolved_input_value(
    shader_input: Any,
    *,
    label: str,
    material_path: str,
) -> tuple[Any, str]:
    """Resolve a value through Material-interface input connections."""

    from pxr import UsdShade

    if not shader_input:
        raise _contract_error(
            "MATERIAL_INPUT_MISSING",
            f"Required material input is missing: {label}.",
            material_path=material_path,
            details={"port": label},
        )

    current = shader_input
    color_space = ""
    visited: set[str] = set()
    for _depth in range(8):
        attr = current.GetAttr()
        attr_path = str(attr.GetPath())
        if attr_path in visited:
            raise _contract_error(
                "MATERIAL_INPUT_CONNECTION_CYCLE",
                f"Material input connection cycle detected at {label}.",
                material_path=material_path,
                details={"port": label, "attribute": attr_path},
            )
        visited.add(attr_path)
        color_space = _input_color_space(current) or color_space

        try:
            sources, invalid_sources = current.GetConnectedSources()
        except Exception as exc:
            raise _contract_error(
                "MATERIAL_CONNECTION_INVALID",
                f"Could not inspect material input {label}: {exc}",
                material_path=material_path,
                details={"port": label},
            ) from exc
        if invalid_sources:
            raise _contract_error(
                "MATERIAL_INPUT_CONNECTION_INVALID",
                f"Material input {label} has invalid authored connections.",
                material_path=material_path,
                details={
                    "port": label,
                    "source_count": len(sources),
                    "invalid_source_paths": [str(path) for path in invalid_sources],
                },
            )
        if not sources:
            return current.Get(), color_space
        if len(sources) != 1 or sources[0].sourceType != UsdShade.AttributeType.Input:
            raise _contract_error(
                "MATERIAL_INPUT_CONNECTION_INVALID",
                f"Material input {label} must resolve through a single input source.",
                material_path=material_path,
                details={"port": label, "source_count": len(sources)},
            )
        current = sources[0].source.GetInput(sources[0].sourceName)
        if not current:
            raise _contract_error(
                "MATERIAL_INPUT_CONNECTION_INVALID",
                f"Material input {label} resolves to a missing interface input.",
                material_path=material_path,
                details={"port": label},
            )

    raise _contract_error(
        "MATERIAL_INPUT_CONNECTION_DEPTH_EXCEEDED",
        f"Material input {label} exceeded the supported connection depth.",
        material_path=material_path,
        details={"port": label},
    )


def _validate_portable_texture_path(
    texture_path: str,
    *,
    library_path: Path,
    material_path: str,
    channel: str,
) -> Path:
    if not texture_path:
        raise _contract_error(
            "MATERIAL_TEXTURE_REFERENCE_MISSING",
            f"The {channel} texture reference is empty.",
            material_path=material_path,
            details={"channel": channel},
        )
    if is_uri_asset_path(texture_path) or is_absolute_asset_path(texture_path):
        raise _contract_error(
            "MATERIAL_TEXTURE_REFERENCE_NOT_PORTABLE",
            f"The {channel} texture must use a package-relative asset path.",
            material_path=material_path,
            details={"channel": channel, "texture_path": texture_path},
        )
    try:
        resolved = resolve_relative_asset_path_under_base(
            texture_path,
            library_path.resolve().parent,
        )
    except ValueError as exc:
        raise _contract_error(
            "MATERIAL_TEXTURE_REFERENCE_NOT_PORTABLE",
            f"The {channel} texture escapes the material package.",
            material_path=material_path,
            details={"channel": channel, "texture_path": texture_path},
        ) from exc
    if not resolved.is_file():
        raise _contract_error(
            "MATERIAL_TEXTURE_REFERENCE_UNRESOLVED",
            f"The {channel} texture reference does not resolve inside the package.",
            material_path=material_path,
            details={"channel": channel, "texture_path": texture_path},
        )
    return resolved


def _validate_materialx_texcoord(
    image_shader: Any,
    *,
    material_path: str,
    channel: str,
) -> None:
    texcoord_shader, output_name = _connected_shader(
        image_shader.GetInput("texcoord"),
        label=f"{channel}.texcoord",
        material_path=material_path,
    )
    _require_source_output(
        output_name,
        label=f"{channel}.texcoord",
        material_path=material_path,
    )
    texcoord_id = _shader_id(texcoord_shader)
    if texcoord_id == "ND_texcoord_vector2":
        index_input = texcoord_shader.GetInput("index")
        if not index_input or index_input.Get() != 0:
            raise _contract_error(
                "OPENPBR_UV_BINDING_INVALID",
                f"The {channel} texture must use MaterialX texcoord index 0 (st).",
                material_path=material_path,
                details={"channel": channel, "texcoord_shader": texcoord_id},
            )
        return
    if texcoord_id == "ND_geompropvalue_vector2":
        geomprop_input = texcoord_shader.GetInput("geomprop")
        if geomprop_input and str(geomprop_input.Get()) == "st":
            return
    raise _contract_error(
        "OPENPBR_UV_BINDING_INVALID",
        f"The {channel} texture must read the st UV primvar.",
        material_path=material_path,
        details={"channel": channel, "texcoord_shader": texcoord_id},
    )


def _validate_materialx_image(
    image_shader: Any,
    *,
    expected_path: str,
    expected_color_space: str,
    library_path: Path,
    material_path: str,
    channel: str,
) -> dict[str, Any]:
    image_id = _shader_id(image_shader)
    if image_id not in _OPENPBR_IMAGE_SHADER_IDS:
        raise _contract_error(
            "OPENPBR_TEXTURE_NODE_INVALID",
            f"The {channel} input must be driven by a MaterialX image node.",
            material_path=material_path,
            details={"channel": channel, "shader_id": image_id},
        )

    value, color_space = _resolved_input_value(
        image_shader.GetInput("file"),
        label=f"{channel}.file",
        material_path=material_path,
    )
    actual_path = value.path if hasattr(value, "path") else str(value or "")
    if actual_path != expected_path:
        raise _contract_error(
            "OPENPBR_TEXTURE_REFERENCE_MISMATCH",
            f"The {channel} graph does not reference the generated texture.",
            material_path=material_path,
            details={
                "channel": channel,
                "expected_path": expected_path,
                "actual_path": actual_path,
            },
        )
    _validate_portable_texture_path(
        actual_path,
        library_path=library_path,
        material_path=material_path,
        channel=channel,
    )
    if color_space.lower() != expected_color_space.lower():
        raise _contract_error(
            "OPENPBR_TEXTURE_COLOR_SPACE_INVALID",
            f"The {channel} texture must use {expected_color_space} color space.",
            material_path=material_path,
            details={
                "channel": channel,
                "expected_color_space": expected_color_space,
                "actual_color_space": color_space,
            },
        )
    _validate_materialx_texcoord(
        image_shader,
        material_path=material_path,
        channel=channel,
    )
    return {
        "asset_path": actual_path,
        "color_space": color_space,
        "uv_primvar": "st",
    }


def _surface_shader(
    material: Any,
    render_context: str,
    *,
    material_path: str,
) -> tuple[Any, str] | None:
    output = material.GetSurfaceOutput(render_context)
    if not output or not output.HasConnectedSource():
        return None
    return _connected_shader(
        output,
        label=f"outputs:{render_context + ':' if render_context else ''}surface",
        material_path=material_path,
    )


def _is_omnipbr_mdl_shader(shader: Any) -> bool:
    source_asset = shader.GetSourceAsset("mdl")
    source_path = source_asset.path if hasattr(source_asset, "path") else ""
    return source_path == "OmniPBR.mdl" and (
        str(shader.GetSourceAssetSubIdentifier("mdl")) == "OmniPBR"
    )


def _resolved_material_profile(
    material: Any,
    *,
    material_path: str,
    requested_profile: MaterialProfile,
    has_prototype: bool,
) -> tuple[str, str | None]:
    mtlx_source = _surface_shader(material, "mtlx", material_path=material_path)
    if mtlx_source is not None:
        shader, _output_name = mtlx_source
        shader_id = _shader_id(shader)
        if shader_id != _OPENPBR_MATERIALX_SHADER_ID:
            if requested_profile == "auto" and has_prototype:
                return "prototype_materialx", "outputs:mtlx:surface"
            raise _contract_error(
                "OPENPBR_SURFACE_SHADER_INVALID",
                "The MaterialX surface is not connected to OpenPBR.",
                material_path=material_path,
                details={"shader_id": shader_id},
            )
        return "openpbr_materialx", "outputs:mtlx:surface"

    mdl_source = _surface_shader(material, "mdl", material_path=material_path)
    if mdl_source is not None and _is_omnipbr_mdl_shader(mdl_source[0]):
        return "omnipbr_mdl", "outputs:mdl:surface"

    universal_source = _surface_shader(material, "", material_path=material_path)
    if universal_source is not None:
        if _shader_id(universal_source[0]) == "UsdPreviewSurface":
            return "preview_surface", "outputs:surface"
        if requested_profile == "auto" and has_prototype:
            return "prototype_universal", "outputs:surface"

    if requested_profile == "auto" and has_prototype:
        return "prototype_unresolved", None

    raise _contract_error(
        "MATERIAL_AUTHORITATIVE_SURFACE_MISSING",
        "The material has no supported authoritative surface output.",
        material_path=material_path,
    )


def _validate_openpbr_materialx_graph(
    material: Any,
    *,
    generated: GeneratedMaterial,
    library_path: Path,
) -> dict[str, Any]:
    material_path = str(material.GetPrim().GetPath())
    mtlx_source = _surface_shader(material, "mtlx", material_path=material_path)
    if mtlx_source is None:
        raise _contract_error(
            "OPENPBR_SURFACE_MISSING",
            "The authoritative outputs:mtlx:surface connection is missing.",
            material_path=material_path,
        )
    surface_shader, surface_output_name = mtlx_source
    if (
        _shader_id(surface_shader) != _OPENPBR_MATERIALX_SHADER_ID
        or surface_output_name != "out"
    ):
        raise _contract_error(
            "OPENPBR_SURFACE_SHADER_INVALID",
            "The MaterialX surface must connect to the OpenPBR out output.",
            material_path=material_path,
            details={
                "shader_id": _shader_id(surface_shader),
                "output_name": surface_output_name,
            },
        )

    expected_paths = {
        "albedo": _relative_asset_path(generated.textures.albedo, library_path),
        "normal": _relative_asset_path(generated.textures.normal, library_path),
        "orm": _relative_asset_path(generated.textures.orm, library_path),
    }
    albedo_shader, albedo_output = _connected_shader(
        surface_shader.GetInput("base_color"),
        label="openpbr.base_color",
        material_path=material_path,
    )
    _require_source_output(
        albedo_output,
        label="openpbr.base_color",
        material_path=material_path,
    )
    albedo_evidence = _validate_materialx_image(
        albedo_shader,
        expected_path=expected_paths["albedo"],
        expected_color_space="sRGB",
        library_path=library_path,
        material_path=material_path,
        channel="albedo",
    )

    roughness_shader, roughness_output = _connected_shader(
        surface_shader.GetInput("specular_roughness"),
        label="openpbr.specular_roughness",
        material_path=material_path,
    )
    metallic_shader, metallic_output = _connected_shader(
        surface_shader.GetInput("base_metalness"),
        label="openpbr.base_metalness",
        material_path=material_path,
    )
    if (
        _shader_id(roughness_shader) != "ND_separate3_color3"
        or roughness_shader.GetPrim() != metallic_shader.GetPrim()
        or roughness_output != "outg"
        or metallic_output != "outb"
        or not roughness_shader.GetOutput("outr")
    ):
        raise _contract_error(
            "OPENPBR_ORM_CONNECTION_INVALID",
            "Raw ORM must use R=AO, G=specular roughness, and B=base metalness.",
            material_path=material_path,
            details={
                "roughness_shader_id": _shader_id(roughness_shader),
                "roughness_output": roughness_output,
                "metallic_output": metallic_output,
            },
        )
    orm_shader, orm_output = _connected_shader(
        roughness_shader.GetInput("in"),
        label="openpbr.orm",
        material_path=material_path,
    )
    _require_source_output(
        orm_output,
        label="openpbr.orm",
        material_path=material_path,
    )
    orm_evidence = _validate_materialx_image(
        orm_shader,
        expected_path=expected_paths["orm"],
        expected_color_space="raw",
        library_path=library_path,
        material_path=material_path,
        channel="orm",
    )

    normalmap_shader, normalmap_output = _connected_shader(
        surface_shader.GetInput("geometry_normal"),
        label="openpbr.geometry_normal",
        material_path=material_path,
    )
    if _shader_id(normalmap_shader) != "ND_normalmap" or normalmap_output != "out":
        raise _contract_error(
            "OPENPBR_NORMAL_CONNECTION_INVALID",
            "The OpenGL tangent-space normal must pass through ND_normalmap.",
            material_path=material_path,
            details={
                "shader_id": _shader_id(normalmap_shader),
                "output_name": normalmap_output,
            },
        )
    normal_shader, normal_output = _connected_shader(
        normalmap_shader.GetInput("in"),
        label="openpbr.normalmap.in",
        material_path=material_path,
    )
    _require_source_output(
        normal_output,
        label="openpbr.normalmap.in",
        material_path=material_path,
    )
    normal_evidence = _validate_materialx_image(
        normal_shader,
        expected_path=expected_paths["normal"],
        expected_color_space="raw",
        library_path=library_path,
        material_path=material_path,
        channel="normal",
    )

    preview_source = _surface_shader(material, "", material_path=material_path)
    if (
        preview_source is None
        or _shader_id(preview_source[0]) != "UsdPreviewSurface"
        or preview_source[0].GetPrim().GetName() != "OVRTXPreviewSurface"
    ):
        raise _contract_error(
            "OPENPBR_OVRTX_PREVIEW_FALLBACK_MISSING",
            "OpenPBR output requires the compatibility-only OVRTX PreviewSurface.",
            material_path=material_path,
        )

    return {
        "authoritative_output": "outputs:mtlx:surface",
        "authoritative_shader_id": _OPENPBR_MATERIALX_SHADER_ID,
        "compatibility_outputs": ["outputs:surface"],
        "compatibility_shader_id": "UsdPreviewSurface",
        "textures": {
            "albedo": albedo_evidence,
            "orm": {
                **orm_evidence,
                "packing": "r_occlusion_g_roughness_b_metallic",
                "connections": {
                    "r": "ambient_occlusion_provenance_only",
                    "g": "specular_roughness",
                    "b": "base_metalness",
                },
            },
            "normal": {
                **normal_evidence,
                "normal_convention": "tangent_opengl_positive_y",
                "normalmap_shader_id": "ND_normalmap",
            },
        },
    }


def _material_texture_references(
    material: Any,
    *,
    library_path: Path,
) -> tuple[str, ...]:
    from pxr import Sdf, Usd

    material_path = str(material.GetPrim().GetPath())
    references: set[str] = set()
    for prim in Usd.PrimRange(material.GetPrim()):
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            raw_path = value.path if hasattr(value, "path") else str(value or "")
            if not raw_path or Path(raw_path).suffix.lower() not in _TEXTURE_SUFFIXES:
                continue
            _validate_portable_texture_path(
                raw_path,
                library_path=library_path,
                material_path=material_path,
                channel="texture",
            )
            references.add(raw_path)
    return tuple(sorted(references))


def inspect_material_library_authoring(
    library_path: str | Path,
    materials: list[GeneratedMaterial] | tuple[GeneratedMaterial, ...],
    *,
    material_profile: str | MaterialProfile,
) -> dict[str, Any]:
    """Reopen and validate authored profile, graph, and package texture references."""

    from pxr import Usd, UsdShade

    library_path = Path(library_path)
    requested_profile = normalize_material_profile(material_profile)
    stage = Usd.Stage.Open(str(library_path))
    if stage is None:
        raise MaterialAuthoringContractError(
            "MATERIAL_LIBRARY_STAGE_REOPEN_FAILED",
            f"Authored material library could not be reopened: {library_path}",
            details={
                "library_path": str(library_path),
                "requested_profile": requested_profile,
            },
        )

    material_evidence: dict[str, Any] = {}
    resolved_profiles: set[str] = set()
    texture_references: set[str] = set()
    for generated in materials:
        material_path = generated.binding
        material_prim = stage.GetPrimAtPath(material_path)
        material = UsdShade.Material(material_prim)
        if not material or not material.GetPrim().IsValid():
            raise _contract_error(
                "MATERIAL_PRIM_MISSING",
                "The authored material prim is missing from the reopened stage.",
                material_path=material_path,
            )

        resolved_profile, authoritative_output = _resolved_material_profile(
            material,
            material_path=material_path,
            requested_profile=requested_profile,
            has_prototype=bool(generated.prototype_source),
        )
        if requested_profile != "auto" and resolved_profile != requested_profile:
            raise _contract_error(
                "MATERIAL_PROFILE_RESOLUTION_MISMATCH",
                "The authored material did not resolve to the explicitly requested profile.",
                material_path=material_path,
                details={
                    "requested_profile": requested_profile,
                    "resolved_profile": resolved_profile,
                },
            )

        references = _material_texture_references(material, library_path=library_path)
        expected_paths = {
            _relative_asset_path(generated.textures.albedo, library_path),
            _relative_asset_path(generated.textures.normal, library_path),
            _relative_asset_path(generated.textures.orm, library_path),
        }
        if (requested_profile != "auto" or not generated.prototype_source) and not (
            expected_paths <= set(references)
        ):
            raise _contract_error(
                "MATERIAL_TEXTURE_REFERENCES_INCOMPLETE",
                "The authored material does not reference every generated texture.",
                material_path=material_path,
                details={
                    "expected_paths": sorted(expected_paths),
                    "actual_paths": list(references),
                },
            )

        profile_evidence: dict[str, Any] = {
            "resolved_profile": resolved_profile,
            "authoritative_output": authoritative_output,
            "texture_references": list(references),
        }
        if resolved_profile == "openpbr_materialx" and (
            requested_profile != "auto" or not generated.prototype_source
        ):
            profile_evidence.update(
                _validate_openpbr_materialx_graph(
                    material,
                    generated=generated,
                    library_path=library_path,
                )
            )
        material_evidence[material_path] = profile_evidence
        resolved_profiles.add(resolved_profile)
        texture_references.update(references)

    resolved_profile = (
        next(iter(resolved_profiles)) if len(resolved_profiles) == 1 else "mixed"
    )
    return {
        "requested_profile": requested_profile,
        "resolved_profile": resolved_profile,
        "resolved_profiles": sorted(resolved_profiles),
        "stage_reopened": True,
        "texture_reference_count": len(texture_references),
        "texture_references": sorted(texture_references),
        "materials": material_evidence,
    }


def write_material_library_usd(
    library_path: str | Path,
    materials: list[GeneratedMaterial] | tuple[GeneratedMaterial, ...],
    *,
    material_profile: str | MaterialProfile = "auto",
    authoring_evidence: dict[str, Any] | None = None,
) -> Path:
    """Write and validate a USD library, optionally returning evidence by mutation."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from world_understanding.utils.usd.material import (
        add_ovrtx_preview_fallbacks_for_materialx_openpbr,
    )

    if not materials:
        raise ValueError("at least one generated material is required")
    material_profile = normalize_material_profile(material_profile)
    require_material_authoring_prerequisites(material_profile)
    if material_profile == "display_color":
        raise ValueError(
            "material_profile='display_color' is apply-only; generated material "
            "libraries do not have target scene prims for displayColor authoring."
        )

    library_path = Path(library_path)
    library_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(library_path))
    UsdGeom.Scope.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, DEFAULT_LIBRARY_ROOT)

    for generated in materials:
        if material_profile == "auto" and _try_author_from_prototype(
            stage,
            library_path,
            generated,
        ):
            continue

        recipe = generated.recipe
        roughness = _optical_roughness(generated, float(recipe.pbr_hints.roughness))
        material_path = recipe.binding
        if material_profile == "openpbr_materialx":
            _define_openpbr_materialx_from_recipe(
                stage,
                library_path,
                generated,
                material_path,
                roughness,
                Sdf,
            )
            continue

        if material_profile == "omnipbr_mdl":
            _define_omnipbr_mdl_material_from_recipe(
                stage,
                library_path,
                generated,
                material_path,
                roughness,
                Sdf,
            )
            continue

        _define_preview_material_from_recipe(
            stage,
            library_path,
            generated,
            material_path,
            roughness,
            Sdf,
        )
        preview = UsdShade.Shader(
            stage.GetPrimAtPath(f"{material_path}/PreviewSurface")
        )

        st_reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader_st")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        albedo = _define_texture_shader(
            stage,
            f"{material_path}/AlbedoTexture",
            _relative_asset_path(generated.textures.albedo, library_path),
            "auto",
            (
                recipe.base_color_hint[0],
                recipe.base_color_hint[1],
                recipe.base_color_hint[2],
                1.0,
            ),
        )
        _connect_texture_st(albedo, st_reader, Sdf)
        diffuse_input = preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        diffuse_input.Set(Gf.Vec3f(*_srgb_color_to_linear(recipe.base_color_hint)))
        diffuse_input.ConnectToSource(
            albedo.ConnectableAPI(),
            "rgb",
        )

        normal = _define_texture_shader(
            stage,
            f"{material_path}/NormalTexture",
            _relative_asset_path(generated.textures.normal, library_path),
            "raw",
            (0.5, 0.5, 1.0, 1.0),
        )
        normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
        )
        normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
        )
        _connect_texture_st(normal, st_reader, Sdf)
        preview.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            normal.ConnectableAPI(),
            "rgb",
        )

        orm = _define_texture_shader(
            stage,
            f"{material_path}/OrmTexture",
            _relative_asset_path(generated.textures.orm, library_path),
            "raw",
            (
                1.0,
                roughness,
                recipe.pbr_hints.metallic,
                1.0,
            ),
        )
        _connect_texture_st(orm, st_reader, Sdf)
        preview.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
            orm.ConnectableAPI(),
            "r",
        )
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            orm.ConnectableAPI(),
            "g",
        )
        preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            orm.ConnectableAPI(),
            "b",
        )

    add_ovrtx_preview_fallbacks_for_materialx_openpbr(stage)
    stage.GetRootLayer().Save()
    evidence = inspect_material_library_authoring(
        library_path,
        materials,
        material_profile=material_profile,
    )
    if authoring_evidence is not None:
        authoring_evidence.clear()
        authoring_evidence.update(evidence)
    return library_path
