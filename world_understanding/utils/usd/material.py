# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD Material utilities for creating and binding MDL materials."""

import logging
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urlparse

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from world_understanding.utils.archive import ArchiveSizeLimitExceeded
from world_understanding.utils.usd.package import (
    extract_usdz_member_to_path,
    package_member_cache_name,
    parse_package_member_asset_path,
    resolve_local_package_path,
)

logger = logging.getLogger(__name__)
_NON_LOCAL_ASSET_SCHEMES = frozenset({"http", "https", "data"})
_OVRTX_PREVIEW_FALLBACK_SHADER_NAME = "OVRTXPreviewSurface"
_OVRTX_PREVIEW_ALBEDO_TEXTURE_NAME = "OVRTXPreviewAlbedoTexture"
_OVRTX_PREVIEW_DISPLAY_COLOR_READER_NAME = "OVRTXPreviewDisplayColorReader"
_OVRTX_PREVIEW_ST_READER_NAME = "OVRTXPreviewSTReader"
_MATERIALX_OPENPBR_SHADER_ID = "ND_open_pbr_surface_surfaceshader"
_OPENPBR_DEFAULT_BASE_COLOR = (0.8, 0.8, 0.8)
_OPENPBR_FULL_TRANSMISSION_PREVIEW_OPACITY = 0.35
_OPENPBR_TRANSMISSION_PREVIEW_THRESHOLD = 0.5
_MDL_TEXTURE_INPUT_NAMES = frozenset(
    {
        "diffuse_texture",
        "albedo_texture",
        "base_color_texture",
        "diffuse_color_texture",
        "detail_normalmap_texture",
        "normalmap_texture",
        "normal_texture",
        "normal_map_texture",
        "orm_texture",
        "reflectionroughness_texture",
        "roughness_texture",
        "specular_roughness_texture",
        "metallic_texture",
        "metalness_texture",
    }
)


def _output_has_connected_source(output: UsdShade.Output) -> bool:
    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return False
    return bool(sources)


def _material_has_connected_surface(
    material: UsdShade.Material,
    render_context: str = "",
) -> bool:
    output = material.GetSurfaceOutput(render_context)
    return bool(output and _output_has_connected_source(output))


def _material_has_connected_texture_capable_mdl_surface(
    material: UsdShade.Material,
) -> bool:
    output = material.GetSurfaceOutput("mdl")
    if not output:
        return False

    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return False

    for source_info in sources:
        source = source_info.source
        if not source:
            continue
        prim = source.GetPrim()
        if not prim or not prim.IsA(UsdShade.Shader):
            continue
        if _shader_has_texture_capable_input(UsdShade.Shader(prim), visited=set()):
            return True
    return False


def _normalized_asset_path_key(path: str) -> str:
    normalized = unquote(path.strip()).replace("\\", "/")
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    if parsed.scheme and parsed.scheme in _NON_LOCAL_ASSET_SCHEMES:
        return normalized
    return os.path.normpath(normalized).replace("\\", "/")


def _asset_path_keys(value: object | None) -> set[str]:
    paths: list[str] = []
    if isinstance(value, Sdf.AssetPath):
        paths.extend(
            [
                value.path,
                str(getattr(value, "resolvedPath", "") or ""),
            ]
        )
    elif isinstance(value, str):
        paths.append(value)

    return {key for key in (_normalized_asset_path_key(path) for path in paths) if key}


def _asset_path_values_match(value: object | None, target: Sdf.AssetPath) -> bool:
    return bool(_asset_path_keys(value) & _asset_path_keys(target))


def _shader_uses_texture_asset(
    shader: UsdShade.Shader,
    texture_asset: Sdf.AssetPath,
    *,
    visited: set[str],
) -> bool:
    prim = shader.GetPrim()
    prim_path = str(prim.GetPath()) if prim else ""
    if prim_path in visited:
        return False
    visited.add(prim_path)

    shader_id_attr = shader.GetIdAttr()
    if shader_id_attr and shader_id_attr.Get() == "UsdUVTexture":
        file_input = shader.GetInput("file")
        if file_input and _asset_path_values_match(file_input.Get(), texture_asset):
            return True

    for inp in shader.GetInputs():
        if inp.GetBaseName().lower() in _MDL_TEXTURE_INPUT_NAMES:
            if _asset_path_values_match(inp.Get(), texture_asset):
                return True

        try:
            sources, _ = inp.GetConnectedSources()
        except Exception:
            continue
        for source_info in sources:
            source = source_info.source
            if not source:
                continue
            source_prim = source.GetPrim()
            if not source_prim or not source_prim.IsA(UsdShade.Shader):
                continue
            if _shader_uses_texture_asset(
                UsdShade.Shader(source_prim),
                texture_asset,
                visited=visited,
            ):
                return True
    return False


def _connected_mdl_surface_uses_texture_asset(
    material: UsdShade.Material,
    texture_asset: Sdf.AssetPath,
) -> bool:
    output = material.GetSurfaceOutput("mdl")
    if not output:
        return False

    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return False

    for source_info in sources:
        source = source_info.source
        if not source:
            continue
        prim = source.GetPrim()
        if not prim or not prim.IsA(UsdShade.Shader):
            continue
        if _shader_uses_texture_asset(
            UsdShade.Shader(prim),
            texture_asset,
            visited=set(),
        ):
            return True
    return False


def _shader_has_texture_capable_input(
    shader: UsdShade.Shader,
    *,
    visited: set[str],
) -> bool:
    prim = shader.GetPrim()
    prim_path = str(prim.GetPath()) if prim else ""
    if prim_path in visited:
        return False
    visited.add(prim_path)

    for inp in shader.GetInputs():
        if inp.GetBaseName().lower() in _MDL_TEXTURE_INPUT_NAMES:
            if _output_has_connected_source(inp):
                return True
            value = inp.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                return True
            if isinstance(value, str) and value.strip():
                return True

        try:
            sources, _ = inp.GetConnectedSources()
        except Exception:
            continue
        for source_info in sources:
            source = source_info.source
            if not source:
                continue
            source_prim = source.GetPrim()
            if not source_prim or not source_prim.IsA(UsdShade.Shader):
                continue
            if _shader_has_texture_capable_input(
                UsdShade.Shader(source_prim),
                visited=visited,
            ):
                return True
    return False


def _connected_materialx_openpbr_surface(
    material: UsdShade.Material,
) -> UsdShade.Shader | None:
    output = material.GetSurfaceOutput("mtlx")
    if not output:
        return None

    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return None

    for source_info in sources:
        source = source_info.source
        if not source:
            continue
        shader = UsdShade.Shader(source.GetPrim())
        if not shader:
            continue
        shader_id_attr = shader.GetIdAttr()
        if shader_id_attr and shader_id_attr.Get() == _MATERIALX_OPENPBR_SHADER_ID:
            return shader
    return None


def _float_material_input(
    material_prim: Usd.Prim,
    input_name: str,
    default: float,
) -> float:
    attr = material_prim.GetAttribute(f"inputs:{input_name}")
    if not attr:
        return default
    value = attr.Get()
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _material_input_value(material_prim: Usd.Prim, input_name: str) -> object | None:
    attr = material_prim.GetAttribute(f"inputs:{input_name}")
    if not attr:
        return None
    value: object | None = attr.Get()
    return value


def _asset_material_input(
    material_prim: Usd.Prim,
    input_name: str,
) -> Sdf.AssetPath | None:
    value = _material_input_value(material_prim, input_name)
    if isinstance(value, Sdf.AssetPath) and value.path:
        return value
    return None


def _mdl_shader_input_asset(
    material_prim: Usd.Prim,
    input_name: str,
) -> Sdf.AssetPath | None:
    for child in material_prim.GetChildren():
        if not child.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(child)
        inp = shader.GetInput(input_name)
        if not inp:
            continue
        value = inp.Get()
        if isinstance(value, Sdf.AssetPath) and value.path:
            return value
    return None


def _mdl_shader_input_value(
    material_prim: Usd.Prim,
    input_name: str,
) -> object | None:
    for child in material_prim.GetChildren():
        if not child.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(child)
        inp = shader.GetInput(input_name)
        if inp:
            return inp.Get()
    return None


def _shader_input_asset(
    shader: UsdShade.Shader,
    input_name: str,
) -> Sdf.AssetPath | None:
    inp = shader.GetInput(input_name)
    if not inp:
        return None
    value = inp.Get()
    if isinstance(value, Sdf.AssetPath) and value.path:
        return value
    return None


def _input_connected_shader_asset(
    shader: UsdShade.Shader,
    input_name: str,
    *,
    visited: set[str] | None = None,
) -> Sdf.AssetPath | None:
    inp = shader.GetInput(input_name)
    if not inp:
        return None

    try:
        sources, _ = inp.GetConnectedSources()
    except Exception:
        return None

    visited = visited or set()
    for source_info in sources:
        source = source_info.source
        if not source:
            continue
        source_prim = source.GetPrim()
        source_path = str(source_prim.GetPath())
        if source_path in visited:
            continue
        visited.add(source_path)

        source_shader = UsdShade.Shader(source_prim)
        shader_id_attr = source_shader.GetIdAttr()
        shader_id = shader_id_attr.Get() if shader_id_attr else None
        if shader_id == "UsdUVTexture":
            texture_asset = _shader_input_asset(source_shader, "file")
            if texture_asset is not None:
                return texture_asset

        texture_asset = _input_connected_shader_asset(
            source_shader,
            source_info.sourceName,
            visited=visited,
        )
        if texture_asset is not None:
            return texture_asset
    return None


def _connected_preview_diffuse_texture_asset(
    material_prim: Usd.Prim,
) -> Sdf.AssetPath | None:
    material = UsdShade.Material(material_prim)
    output = material.GetSurfaceOutput()
    if not output:
        return None

    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return None

    for source_info in sources:
        source = source_info.source
        if not source:
            continue
        shader = UsdShade.Shader(source.GetPrim())
        texture_asset = _input_connected_shader_asset(shader, "diffuseColor")
        if texture_asset is not None:
            return texture_asset
    return None


def _preview_base_color_texture_asset(material_prim: Usd.Prim) -> Sdf.AssetPath | None:
    connected_preview = _connected_preview_diffuse_texture_asset(material_prim)
    return (
        _asset_material_input(material_prim, "base_color_texture_file")
        or _asset_material_input(material_prim, "diffuse_texture")
        or _mdl_shader_input_asset(material_prim, "diffuse_texture")
        or connected_preview
    )


def _openpbr_preview_diffuse_color(
    material_prim: Usd.Prim,
    transmission_weight: float,
) -> object:
    if transmission_weight >= _OPENPBR_TRANSMISSION_PREVIEW_THRESHOLD:
        transmission_color = _material_input_value(
            material_prim,
            "transmission_color",
        )
        if transmission_color is not None:
            return transmission_color

    base_color = _material_input_value(material_prim, "base_color")
    if base_color is not None:
        return base_color
    return _OPENPBR_DEFAULT_BASE_COLOR


def _openpbr_preview_opacity(
    material_prim: Usd.Prim,
    transmission_weight: float,
) -> float:
    geometry_opacity = _float_material_input(material_prim, "geometry_opacity", 1.0)
    if transmission_weight < _OPENPBR_TRANSMISSION_PREVIEW_THRESHOLD:
        return geometry_opacity

    # OpenPBR glass remains geometrically opaque while becoming optically
    # transmissive. UsdPreviewSurface has no transmission input, so approximate
    # transmissive materials with alpha for render-only OVRTX exports.
    transmission_alpha = (
        1.0 - (1.0 - _OPENPBR_FULL_TRANSMISSION_PREVIEW_OPACITY) * transmission_weight
    )
    return max(0.0, min(1.0, geometry_opacity * transmission_alpha))


def _material_path_filter(
    target_material_paths: Iterable[str | Sdf.Path] | None,
) -> set[str] | None:
    if target_material_paths is None:
        return None
    return {str(path) for path in target_material_paths}


def _iter_materialx_openpbr_fallback_prims(
    stage: Usd.Stage,
    target_material_paths: set[str] | None = None,
) -> list[Usd.Prim]:
    fallback_prims: list[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        if (
            target_material_paths is not None
            and str(prim.GetPath()) not in target_material_paths
        ):
            continue

        material = UsdShade.Material(prim)
        if _material_has_connected_surface(material) or _material_has_connected_surface(
            material,
            "mdl",
        ):
            continue
        if _connected_materialx_openpbr_surface(material) is None:
            continue
        fallback_prims.append(prim)
    return fallback_prims


def _iter_materialx_openpbr_surface_prims(
    stage: Usd.Stage,
    target_material_paths: set[str] | None = None,
) -> list[Usd.Prim]:
    surface_prims: list[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        if (
            target_material_paths is not None
            and str(prim.GetPath()) not in target_material_paths
        ):
            continue
        if _connected_materialx_openpbr_surface(UsdShade.Material(prim)) is None:
            continue
        surface_prims.append(prim)
    return surface_prims


def _prepare_material_for_surface_authoring(material: UsdShade.Material) -> bool:
    prim = material.GetPrim()
    if not prim or not prim.IsValid():
        return False
    if prim.IsInstanceProxy():
        return False
    if prim.IsInstance() or prim.IsInstanceable():
        prim.SetInstanceable(False)
    return True


def _suppress_materialx_surface(material: UsdShade.Material) -> bool:
    if not _prepare_material_for_surface_authoring(material):
        return False
    material.CreateSurfaceOutput("mtlx").GetAttr().SetConnections([])
    return True


def _author_ovrtx_preview_fallback(
    target_stage: Usd.Stage,
    material_path: str,
    source_material_prim: Usd.Prim,
    *,
    suppress_materialx_surface: bool = False,
) -> bool:
    source_is_instance = (
        source_material_prim.IsInstance() or source_material_prim.IsInstanceable()
    )
    target_prim = target_stage.GetPrimAtPath(material_path)
    if target_prim.IsValid():
        if target_prim.IsInstanceProxy():
            return False
        if target_prim.IsInstance() or target_prim.IsInstanceable():
            target_prim.SetInstanceable(False)

    material = UsdShade.Material.Define(target_stage, material_path)
    if source_is_instance:
        material.GetPrim().SetInstanceable(False)
    if not _prepare_material_for_surface_authoring(material):
        return False

    shader = UsdShade.Shader.Define(
        target_stage,
        f"{material_path}/{_OVRTX_PREVIEW_FALLBACK_SHADER_NAME}",
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

    transmission_weight = _float_material_input(
        source_material_prim,
        "transmission_weight",
        0.0,
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        _openpbr_preview_diffuse_color(source_material_prim, transmission_weight),
    )

    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "base_metalness", 0.0),
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "specular_roughness", 0.5),
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
        _openpbr_preview_opacity(source_material_prim, transmission_weight),
    )

    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )

    if suppress_materialx_surface:
        # OVRTX can prefer the MaterialX render context over the universal
        # UsdPreviewSurface output and fall back to its red error shader.
        _suppress_materialx_surface(material)
    return True


def _coerce_color3f(value: object | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        components = list(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(components) < 3:
        return None
    try:
        color = tuple(max(0.0, min(1.0, float(v))) for v in components[:3])
    except (TypeError, ValueError):
        return None
    return color  # type: ignore[return-value]


def _stage_asset_base_dir(stage: Usd.Stage) -> Path:
    root_layer = stage.GetRootLayer()
    if root_layer.realPath:
        return Path(root_layer.realPath).parent
    return Path.cwd()


def _texture_asset_sampling_candidates(
    texture_asset: Sdf.AssetPath,
    *,
    base_dir: Path | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    resolved_path = str(getattr(texture_asset, "resolvedPath", "") or "")
    if resolved_path:
        candidates.append(Path(resolved_path))
    if texture_asset.path and os.path.isabs(texture_asset.path):
        candidates.append(Path(texture_asset.path))
    elif texture_asset.path and base_dir is not None:
        candidates.append(base_dir / texture_asset.path)
    return candidates


def _sample_texture_average_color(
    texture_asset: Sdf.AssetPath,
    *,
    base_dir: Path | None = None,
) -> tuple[float, float, float] | None:
    for candidate in _texture_asset_sampling_candidates(
        texture_asset,
        base_dir=base_dir,
    ):
        if not _safe_exists(candidate):
            continue
        try:
            from PIL import Image

            with Image.open(candidate) as image:
                pixel = (
                    image.convert("RGB")
                    .resize((1, 1), Image.Resampling.BOX)
                    .getpixel((0, 0))
                )
            return tuple(channel / 255.0 for channel in pixel)
        except Exception as exc:
            logger.debug(
                "Failed to sample preview color from %s: %s",
                candidate,
                exc,
            )
    return None


def _preview_base_color_value(
    material_prim: Usd.Prim,
    albedo_texture: Sdf.AssetPath,
    *,
    base_dir: Path | None = None,
) -> tuple[float, float, float]:
    return (
        _sample_texture_average_color(albedo_texture, base_dir=base_dir)
        or _coerce_color3f(_material_input_value(material_prim, "base_color"))
        or _coerce_color3f(_mdl_shader_input_value(material_prim, "diffuse_tint"))
        or _OPENPBR_DEFAULT_BASE_COLOR
    )


def _sample_texture_at_uv(
    image: object,
    uv: object,
) -> tuple[float, float, float] | None:
    try:
        u = float(uv[0])  # type: ignore[index]
        v = float(uv[1])  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(u) or not math.isfinite(v):
        return None

    width, height = image.size  # type: ignore[attr-defined]
    if width <= 0 or height <= 0:
        return None

    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    x = min(width - 1, max(0, round(u * (width - 1))))
    y = min(height - 1, max(0, round((1.0 - v) * (height - 1))))
    try:
        pixel = image.getpixel((x, y))  # type: ignore[attr-defined]
    except Exception:
        return None
    return tuple(float(channel) / 255.0 for channel in pixel[:3])


def _open_texture_image(
    texture_asset: Sdf.AssetPath,
    *,
    base_dir: Path | None = None,
) -> object | None:
    for candidate in _texture_asset_sampling_candidates(
        texture_asset,
        base_dir=base_dir,
    ):
        if not _safe_exists(candidate):
            continue
        try:
            from PIL import Image

            with Image.open(candidate) as image:
                return image.convert("RGB").copy()
        except Exception as exc:
            logger.debug(
                "Failed to open texture for displayColor bake from %s: %s",
                candidate,
                exc,
            )
    return None


def _expanded_primvar_values(
    values: object,
    indices: object,
    expected_count: int,
) -> list[object] | None:
    try:
        value_list = list(values)  # type: ignore[arg-type]
    except TypeError:
        return None

    if expected_count == 0:
        return []

    try:
        index_list = list(indices)  # type: ignore[arg-type]
    except TypeError:
        index_list = []

    if len(index_list) == expected_count:
        expanded: list[object] = []
        for index in index_list:
            try:
                expanded.append(value_list[int(index)])
            except (IndexError, TypeError, ValueError):
                return None
        return expanded

    if len(value_list) == expected_count:
        return value_list
    return None


def _mesh_uv_values_for_display_color(
    mesh: UsdGeom.Mesh,
) -> tuple[str, list[object]] | None:
    prim = mesh.GetPrim()
    st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if not st or not st.HasValue():
        return None

    values = st.Get()
    if values is None:
        return None

    interpolation = str(st.GetInterpolation() or "faceVarying")
    indices = st.GetIndices()
    face_counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    point_count = len(mesh.GetPointsAttr().Get() or [])

    expected_by_interpolation = {
        "faceVarying": sum(int(count) for count in face_counts),
        "vertex": point_count,
        "varying": point_count,
        "uniform": len(face_counts),
        "constant": 1,
    }
    expected_count = expected_by_interpolation.get(interpolation)
    if expected_count is None:
        return None

    expanded = _expanded_primvar_values(values, indices, expected_count)
    if expanded is None:
        return None
    return interpolation, expanded


def _author_mesh_display_color(
    mesh: UsdGeom.Mesh,
    interpolation: str,
    colors: list[tuple[float, float, float]],
) -> None:
    display_color = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        interpolation,
    )
    display_color.Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
                for color in colors
            ],
        ),
    )


def bake_texture_file_materials_to_display_color_for_render(
    stage: Usd.Stage,
) -> int:
    """Bake textured material albedo into mesh displayColor for render fallback.

    This is intended for render-only flattened stages. It does not change any
    material texture inputs; it authors ``primvars:displayColor`` on meshes
    bound to materials that reference a local albedo texture. Renderers that
    cannot evaluate ``UsdUVTexture`` can still show texture variation through a
    ``UsdPrimvarReader_float3`` preview fallback.

    Returns the number of mesh prims that received displayColor samples.
    """
    base_dir = _stage_asset_base_dir(stage)
    material_textures: dict[str, Sdf.AssetPath] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        albedo_texture = _preview_base_color_texture_asset(prim)
        if albedo_texture is not None:
            material_textures[str(prim.GetPath())] = albedo_texture

    if not material_textures:
        return 0

    updated = 0
    image_cache: dict[str, object] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if prim.IsInstanceProxy():
            continue
        if prim.IsInstance() or prim.IsInstanceable():
            prim.SetInstanceable(False)

        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not material:
            continue

        albedo_texture = material_textures.get(str(material.GetPath()))
        if albedo_texture is None:
            continue

        uv_info = _mesh_uv_values_for_display_color(UsdGeom.Mesh(prim))
        if uv_info is None:
            continue
        interpolation, uv_values = uv_info

        image_key = str(
            getattr(albedo_texture, "resolvedPath", "")
            or albedo_texture.path
            or str(material.GetPath()),
        )
        image = image_cache.get(image_key)
        if image is None:
            image = _open_texture_image(albedo_texture, base_dir=base_dir)
            if image is None:
                continue
            image_cache[image_key] = image

        colors: list[tuple[float, float, float]] = []
        for uv in uv_values:
            color = _sample_texture_at_uv(image, uv)
            if color is None:
                colors = []
                break
            colors.append(color)
        if not colors:
            continue

        _author_mesh_display_color(UsdGeom.Mesh(prim), interpolation, colors)
        updated += 1
    return updated


def _author_ovrtx_textured_preview_fallback(
    target_stage: Usd.Stage,
    material_path: str,
    source_material_prim: Usd.Prim,
    albedo_texture: Sdf.AssetPath,
    *,
    connect_diffuse_texture: bool = False,
    diffuse_color_primvar: str | None = None,
) -> bool:
    source_is_instance = (
        source_material_prim.IsInstance() or source_material_prim.IsInstanceable()
    )
    target_prim = target_stage.GetPrimAtPath(material_path)
    if target_prim.IsValid():
        if target_prim.IsInstanceProxy():
            return False
        if target_prim.IsInstance() or target_prim.IsInstanceable():
            target_prim.SetInstanceable(False)

    material = UsdShade.Material.Define(target_stage, material_path)
    if source_is_instance:
        material.GetPrim().SetInstanceable(False)
    if not _prepare_material_for_surface_authoring(material):
        return False

    shader = UsdShade.Shader.Define(
        target_stage,
        f"{material_path}/{_OVRTX_PREVIEW_FALLBACK_SHADER_NAME}",
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    fallback_color = _preview_base_color_value(
        source_material_prim,
        albedo_texture,
        base_dir=_stage_asset_base_dir(target_stage),
    )
    diffuse_color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if connect_diffuse_texture:
        st_reader = UsdShade.Shader.Define(
            target_stage,
            f"{material_path}/{_OVRTX_PREVIEW_ST_READER_NAME}",
        )
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(0.0, 0.0),
        )

        albedo = UsdShade.Shader.Define(
            target_stage,
            f"{material_path}/{_OVRTX_PREVIEW_ALBEDO_TEXTURE_NAME}",
        )
        albedo.CreateIdAttr("UsdUVTexture")
        albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(albedo_texture)
        albedo.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        albedo.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        albedo.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        albedo.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(
            Gf.Vec4f(
                float(fallback_color[0]),
                float(fallback_color[1]),
                float(fallback_color[2]),
                1.0,
            ),
        )
        albedo.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2),
        )
        diffuse_color.ConnectToSource(
            albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3),
        )
    elif diffuse_color_primvar:
        reader = UsdShade.Shader.Define(
            target_stage,
            f"{material_path}/{_OVRTX_PREVIEW_DISPLAY_COLOR_READER_NAME}",
        )
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(
            diffuse_color_primvar,
        )
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(fallback_color)
        reader_output = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        diffuse_color.ConnectToSource(reader_output)
    else:
        diffuse_color.Set(fallback_color)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "base_metalness", 0.0),
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "specular_roughness", 0.5),
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "geometry_opacity", 1.0),
    )

    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)
    return True


def _material_surface_is_ovrtx_preview_fallback(material: UsdShade.Material) -> bool:
    output = material.GetSurfaceOutput()
    if not output:
        return False
    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return False
    for source_info in sources:
        source = source_info.source
        if source and source.GetPrim().GetName() == _OVRTX_PREVIEW_FALLBACK_SHADER_NAME:
            return True
    return False


def add_ovrtx_preview_fallbacks_for_texture_file_materials(
    stage: Usd.Stage,
    *,
    override_existing_surface: bool = False,
    connect_diffuse_texture: bool = False,
    diffuse_color_primvar: str | None = None,
    skip_connected_mdl_surface: bool = False,
) -> int:
    """Add render-only UsdPreviewSurface fallbacks for textured materials.

    Texture Agent can author generated texture paths on SimReady/MDL-style
    material inputs without a universal ``outputs:surface`` graph. OVRTX may
    then fall back to an error material or fail to evaluate the MDL texture
    graph even though the generated albedo resolves correctly. This creates a
    lightweight universal preview surface. By default it uses a sampled albedo
    color. Set ``connect_diffuse_texture`` to preserve the generated albedo as a
    simple ``UsdUVTexture`` graph, or set ``diffuse_color_primvar`` to read a
    baked color primvar with that sampled color as its fallback. These two
    fallback modes are mutually exclusive. Authored texture inputs and MDL
    outputs are left intact. Set
    ``override_existing_surface`` only for render-only flattened stages where
    OVRTX needs a stronger fallback than the authored preview graph. Set
    ``skip_connected_mdl_surface`` for self-contained OVRTX bundles where MDL
    texture graphs are known to resolve and should keep their richer material
    response.
    """
    if connect_diffuse_texture and diffuse_color_primvar:
        raise ValueError(
            "connect_diffuse_texture and diffuse_color_primvar are mutually exclusive"
        )

    updated = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue

        material = UsdShade.Material(prim)
        if _material_surface_is_ovrtx_preview_fallback(material):
            continue
        albedo_texture = _preview_base_color_texture_asset(prim)
        if albedo_texture is None:
            continue
        if skip_connected_mdl_surface and _connected_mdl_surface_uses_texture_asset(
            material, albedo_texture
        ):
            continue
        if _material_has_connected_surface(material) and not override_existing_surface:
            continue

        if _author_ovrtx_textured_preview_fallback(
            stage,
            str(prim.GetPath()),
            prim,
            albedo_texture,
            connect_diffuse_texture=connect_diffuse_texture,
            diffuse_color_primvar=diffuse_color_primvar,
        ):
            updated += 1
    return updated


def add_ovrtx_preview_fallbacks_for_materialx_openpbr(
    stage: Usd.Stage,
    *,
    suppress_materialx_surface: bool = False,
    target_material_paths: Iterable[str | Sdf.Path] | None = None,
) -> int:
    """Add temporary UsdPreviewSurface fallbacks for OVRTX MaterialX rendering.

    OVRTX can resolve ordinary universal surface outputs and MDL render
    contexts, but the bundled OpenPBR MaterialX library only authors
    ``outputs:mtlx:surface``. For render-only exports, synthesize a lightweight
    preview shader from the material's direct OpenPBR constants and connect it
    to ``outputs:surface``. The original MaterialX network remains intact unless
    ``suppress_materialx_surface`` is enabled for a render-only export.
    ``target_material_paths`` optionally limits the operation to a known set of
    material prim paths; when omitted, the whole stage is processed.

    Returns the number of Material prims that were updated.
    """
    updated = 0
    target_paths = _material_path_filter(target_material_paths)
    fallback_prims = _iter_materialx_openpbr_fallback_prims(stage, target_paths)
    fallback_paths = {str(prim.GetPath()) for prim in fallback_prims}

    for prim in fallback_prims:
        if _author_ovrtx_preview_fallback(
            stage,
            str(prim.GetPath()),
            prim,
            suppress_materialx_surface=suppress_materialx_surface,
        ):
            updated += 1

    if suppress_materialx_surface:
        for prim in _iter_materialx_openpbr_surface_prims(stage, target_paths):
            if str(prim.GetPath()) in fallback_paths:
                continue

            material = UsdShade.Material(prim)
            if not _material_has_connected_surface(material):
                continue

            if _suppress_materialx_surface(material):
                updated += 1

    return updated


def write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
    stage: Usd.Stage,
    overlay_path: str | Path,
    *,
    suppress_materialx_surface: bool = True,
) -> int:
    """Write a stronger overlay with OVRTX preview fallbacks for a composed stage.

    This covers materials that live in sublayers or references while keeping the
    source stage and its authored material libraries untouched. By default the
    overlay also blocks the MaterialX surface connection so OVRTX must use the
    preview fallback instead of its red error shader path.
    """
    fallback_prims = _iter_materialx_openpbr_fallback_prims(stage)
    fallback_paths = {str(prim.GetPath()) for prim in fallback_prims}
    suppress_only_prims: list[Usd.Prim] = []
    if suppress_materialx_surface:
        for prim in _iter_materialx_openpbr_surface_prims(stage):
            if str(prim.GetPath()) in fallback_paths:
                continue

            material = UsdShade.Material(prim)
            if not _material_has_connected_surface(material):
                continue

            suppress_only_prims.append(prim)

    if not fallback_prims and not suppress_only_prims:
        return 0

    path = Path(overlay_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_stage = Usd.Stage.CreateNew(str(path))
    updated = 0
    for prim in fallback_prims:
        if _author_ovrtx_preview_fallback(
            overlay_stage,
            str(prim.GetPath()),
            prim,
            suppress_materialx_surface=suppress_materialx_surface,
        ):
            updated += 1
    for prim in suppress_only_prims:
        material = UsdShade.Material.Define(overlay_stage, str(prim.GetPath()))
        if prim.IsInstance() or prim.IsInstanceable():
            material.GetPrim().SetInstanceable(False)
        if _suppress_materialx_surface(material):
            updated += 1
    overlay_stage.GetRootLayer().Save()
    return updated


def add_ovrtx_preview_fallbacks_to_stage_file(
    stage_path: str | Path,
    *,
    suppress_materialx_surface: bool = True,
) -> int:
    """Open a USD file, add OVRTX preview fallbacks, and save it if changed."""
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        return 0

    added = add_ovrtx_preview_fallbacks_for_materialx_openpbr(
        stage,
        suppress_materialx_surface=suppress_materialx_surface,
    )
    if added:
        stage.GetRootLayer().Save()
    return added


def _path_leaf_name(path: str | Sdf.Path) -> str:
    return Sdf.Path(str(path)).name


def ensure_looks_scope_spec(
    layer: Sdf.Layer,
    prim_path: str | Sdf.Path,
    *,
    allow_over: bool = False,
) -> None:
    """Type an existing untyped ``Looks`` prim spec as ``Scope``.

    The caller owns creating the prim spec and choosing its specifier
    (``def`` vs ``over``); this only authors the schema type when missing and
    does not repair a non-empty non-``Scope`` type. ``over`` specs require
    explicit opt-in because an ``over Scope`` can override a composed type.
    """
    if _path_leaf_name(prim_path) != "Looks":
        return

    prim_spec = layer.GetPrimAtPath(str(prim_path))
    if (
        prim_spec
        and not prim_spec.typeName
        and (allow_over or prim_spec.specifier != Sdf.SpecifierOver)
    ):
        prim_spec.typeName = "Scope"


def _author_looks_scope_type(stage: Usd.Stage, prim_path: Sdf.Path) -> None:
    """Author a ``Scope`` type opinion without changing the prim specifier."""
    layer = stage.GetEditTarget().GetLayer()
    if layer.GetPrimAtPath(str(prim_path)) is None:
        Sdf.CreatePrimInLayer(layer, prim_path)
    ensure_looks_scope_spec(layer, prim_path, allow_over=True)


def ensure_looks_scope(stage: Usd.Stage, material_path: str | Sdf.Path) -> None:
    """Type an untyped ``Looks`` ancestor of a material path as ``Scope``.

    This intentionally normalizes only the conventional material container,
    not arbitrary intermediate grouping prims. Missing ``Looks`` containers are
    created as ``def Scope``; existing untyped prims only receive a ``Scope``
    type opinion in the current edit layer.
    """
    material_path_str = str(material_path)
    if not material_path_str:
        return

    path = Sdf.Path(material_path_str)
    if not path.IsAbsolutePath():
        return

    parent_path = path.GetParentPath()
    while parent_path != Sdf.Path.absoluteRootPath:
        if _path_leaf_name(parent_path) == "Looks":
            parent_prim = stage.GetPrimAtPath(parent_path)
            if not parent_prim.IsValid():
                UsdGeom.Scope.Define(stage, parent_path)
            elif parent_prim.GetTypeName() == "":
                _author_looks_scope_type(stage, parent_path)
        parent_path = parent_path.GetParentPath()


def _resolve_local_asset_path(
    asset_val: object,
    authored_path: str,
    base_dir: Path,
) -> tuple[str | None, bool]:
    """Resolve a local USD asset path, preferring USD's authored-layer result."""
    resolved_path = str(getattr(asset_val, "resolvedPath", "") or "")
    if resolved_path and _safe_exists(resolved_path):
        return str(Path(resolved_path).resolve()), True

    parsed = urlparse(authored_path)
    if parsed.scheme.lower() == "file":
        path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            path = f"//{parsed.netloc}{path}"
        if len(path) >= 4 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        if _safe_exists(path):
            return str(Path(path).resolve()), True
        return None, False

    if os.path.isabs(authored_path):
        if _safe_exists(authored_path):
            return str(Path(authored_path).resolve()), True
        return None, False

    candidate = base_dir / authored_path
    if _safe_exists(candidate):
        return str(candidate.resolve()), True
    return None, False


def _safe_exists(path: str | Path) -> bool:
    """Return whether a path exists without surfacing invalid path errors."""
    try:
        return Path(path).exists()
    except (OSError, ValueError):
        return False


def _is_non_local_asset_uri(asset_path: str) -> bool:
    """Return true for asset URI values that should not be treated as files."""
    return urlparse(asset_path).scheme.lower() in _NON_LOCAL_ASSET_SCHEMES


def get_local_mdl_assets(
    stage: Usd.Stage, base_dir: str | Path | None = None
) -> list[dict]:
    """Get all local MDL sourceAsset paths from the stage.

    This function traverses the stage to find all Shader prims with MDL
    sourceAsset attributes and returns information about each one. It
    resolves paths to determine which are local files that need bundling.

    Args:
        stage: USD stage to scan for MDL materials
        base_dir: Fallback base directory for resolving relative paths when
            USD does not provide ``Sdf.AssetPath.resolvedPath``. If None, uses
            the stage's root layer directory.

    Returns:
        List of dicts, each containing:
            - shader_path: SdfPath string to the shader prim
            - mdl_path: Original MDL path as stored in the attribute
            - resolved_path: Resolved absolute path to MDL file, or None if:
                - Path is a remote URL (http/https)
                - File doesn't exist locally
            - is_local: True if the file exists locally
    """
    if base_dir is None:
        # Use the root layer's directory as base
        root_layer = stage.GetRootLayer()
        if root_layer.realPath:
            base_dir = Path(root_layer.realPath).parent
        else:
            base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    mdl_assets = []

    for prim in stage.Traverse():
        # Check if it's a Shader prim
        if not prim.IsA(UsdShade.Shader):
            continue

        # Look for MDL sourceAsset attribute
        mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
        if not mdl_attr or not mdl_attr.IsValid():
            continue

        asset_val = mdl_attr.Get()
        if asset_val is None:
            continue

        # Get the path from Sdf.AssetPath
        try:
            mdl_path = asset_val.path if hasattr(asset_val, "path") else str(asset_val)
        except Exception:
            mdl_path = str(asset_val)

        if not mdl_path:
            continue

        # Check if it's a remote or embedded URI - skip these
        if _is_non_local_asset_uri(mdl_path):
            mdl_assets.append(
                {
                    "shader_path": str(prim.GetPath()),
                    "mdl_path": mdl_path,
                    "resolved_path": None,
                    "is_local": False,
                }
            )
            continue

        resolved_path, is_local = _resolve_local_asset_path(
            asset_val,
            mdl_path,
            base_dir,
        )

        mdl_assets.append(
            {
                "shader_path": str(prim.GetPath()),
                "mdl_path": mdl_path,
                "resolved_path": resolved_path,
                "is_local": is_local,
            }
        )

    return mdl_assets


def get_unique_mdl_directories(mdl_assets: list[dict]) -> list[Path]:
    """Get unique directories containing local MDL files.

    MDL materials often have texture files in the same directory,
    so we need to copy the entire directory, not just the MDL file.

    Args:
        mdl_assets: List of MDL asset dicts from get_local_mdl_assets()

    Returns:
        List of unique directory Paths containing local MDL files
    """
    directories = set()

    for asset in mdl_assets:
        if asset["is_local"] and asset["resolved_path"]:
            mdl_file = Path(asset["resolved_path"])
            directories.add(mdl_file.parent)

    return list(directories)


# Image file extensions recognized as texture files
_TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".exr", ".tga", ".hdr", ".bmp"}
_MAX_PACKAGE_TEXTURE_BYTES = 512 * 1024 * 1024


def _package_member_asset_parts(
    asset_path: str,
    *,
    base_dir: Path | None = None,
) -> tuple[Path, str] | None:
    return parse_package_member_asset_path(asset_path, base_dir=base_dir)


def _resolve_package_asset_path(package_ref: str, base_dir: Path | None) -> Path:
    return resolve_local_package_path(package_ref, base_dir)


def _localized_package_texture_root(package_path: Path) -> str:
    return package_member_cache_name(package_path, digest_len=12)


def localize_package_texture_assets_for_render(
    stage: Usd.Stage,
    output_dir: str | Path,
) -> int:
    """Extract USDZ package-member texture refs for render-only remote export.

    Flattening a USDZ for the REST renderer can leave asset paths such as
    ``/path/asset.usdz[0/albedo.png]``. A data-URI stage upload does not include
    that package member, so remote OVRTX can resolve the shader graph but not the
    texture image. This extracts image members to ``output_dir`` and rewrites
    matching ``SdfAssetPath`` attributes to ordinary local file paths that the
    existing render bundler can include.
    """
    out_dir = Path(output_dir)
    root_layer_path = stage.GetRootLayer().realPath or stage.GetRootLayer().identifier
    base_dir = Path(root_layer_path).parent if root_layer_path else None
    updated = 0
    extracted: dict[tuple[Path, str], Path] = {}

    for prim in stage.Traverse():
        if prim.IsInstanceProxy():
            continue
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue

            asset_val = attr.Get()
            if asset_val is None:
                continue
            asset_path = getattr(asset_val, "path", "")
            if not asset_path:
                continue

            package_parts = _package_member_asset_parts(
                str(asset_path),
                base_dir=base_dir,
            )
            if package_parts is None:
                continue
            package_path, member = package_parts
            if Path(member).suffix.lower() not in _TEXTURE_EXTENSIONS:
                continue

            key = (package_path.resolve(), member)
            dest = extracted.get(key)
            if dest is None:
                dest = out_dir / _localized_package_texture_root(package_path) / member
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    written = extract_usdz_member_to_path(
                        package_path,
                        member,
                        dest,
                        allowed_suffixes=_TEXTURE_EXTENSIONS,
                        max_bytes=_MAX_PACKAGE_TEXTURE_BYTES,
                    )
                    if written is None:
                        continue
                except ArchiveSizeLimitExceeded:
                    logger.warning(
                        "Skipped USDZ texture %s[%s] because it exceeds %d bytes",
                        package_path,
                        member,
                        _MAX_PACKAGE_TEXTURE_BYTES,
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        "Failed to extract USDZ texture %s[%s]: %s",
                        package_path,
                        member,
                        exc,
                    )
                    continue
                extracted[key] = dest

            if prim.IsInstance() or prim.IsInstanceable():
                prim.SetInstanceable(False)
            attr.Set(Sdf.AssetPath(str(dest.resolve())))
            updated += 1

    return updated


def get_local_texture_file_assets(
    stage: Usd.Stage, base_dir: str | Path | None = None
) -> list[dict]:
    """Get all local texture file asset paths from the stage.

    This function traverses the stage to find all prims with Sdf.AssetPath-typed
    attributes pointing to image files (PNG, JPG, EXR, TGA, HDR, BMP). It catches
    both direct ``inputs:file`` on UsdUVTexture shaders and texture paths on
    Material prims (e.g. ``inputs:DiffuseTexture``) — important because after
    ``duplicate_stage()`` flattening, paths may live on Material prims.

    Args:
        stage: USD stage to scan for texture references
        base_dir: Fallback base directory for resolving relative paths when
            USD does not provide ``Sdf.AssetPath.resolvedPath``. If None, uses
            the stage's root layer directory.

    Returns:
        List of dicts (deduplicated by resolved_path), each containing:
            - prim_path: SdfPath string to the prim
            - attr_name: Name of the attribute containing the texture path
            - file_path: Original file path as stored in the attribute
            - resolved_path: Resolved absolute path to the texture file, or None
            - is_local: True if the file exists locally
    """
    if base_dir is None:
        root_layer = stage.GetRootLayer()
        if root_layer.realPath:
            base_dir = Path(root_layer.realPath).parent
        else:
            base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    texture_assets: list[dict] = []
    seen_resolved: set[str] = set()

    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            type_name = attr.GetTypeName()
            if type_name.type.typeName != "SdfAssetPath":
                continue

            asset_val = attr.Get()
            if asset_val is None:
                continue

            try:
                file_path = (
                    asset_val.path if hasattr(asset_val, "path") else str(asset_val)
                )
            except Exception:
                file_path = str(asset_val)

            if not file_path:
                continue

            # Skip remote or embedded URIs before treating the value as a path.
            if _is_non_local_asset_uri(file_path):
                texture_assets.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "attr_name": attr.GetName(),
                        "file_path": file_path,
                        "resolved_path": None,
                        "is_local": False,
                    }
                )
                continue

            # Check if extension is a known texture format
            ext = Path(file_path).suffix.lower()
            if ext not in _TEXTURE_EXTENSIONS:
                continue

            resolved_path, is_local = _resolve_local_asset_path(
                asset_val,
                file_path,
                base_dir,
            )

            # Deduplicate by resolved_path
            if resolved_path and resolved_path in seen_resolved:
                continue
            if resolved_path:
                seen_resolved.add(resolved_path)

            texture_assets.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "attr_name": attr.GetName(),
                    "file_path": file_path,
                    "resolved_path": resolved_path,
                    "is_local": is_local,
                }
            )

    return texture_assets


def add_mdl_material(
    stage: Usd.Stage,
    material_name: str,
    source_asset_path: str,
    sub_identifier: str = "OmniSurface",
    path_prefix: str = None,
    color: str | None = None,
) -> tuple[Usd.Stage, str]:
    """Add MDL material to a USD stage.

    Args:
        stage: The USD stage to add the material to
        material_name: Name for the material prim (should be sanitized for use as USD prim name,
                      with spaces, slashes, and dashes replaced with underscores)
        source_asset_path: Path to the MDL source asset
        sub_identifier: MDL subidentifier (typically the material name within the MDL)
        path_prefix: Optional path prefix for the material location (defaults to DefaultPrim/Looks)
        color: Optional hex color value for material modification (not yet implemented)

    Returns:
        Tuple of (updated stage, material_path)
    """
    if not path_prefix:
        default_prim = stage.GetDefaultPrim()
        if default_prim.IsValid():
            path_prefix = str(default_prim.GetPath())
        else:
            # Default prim is invalid (not set or stale after optimization).
            # Fall back to the first root-level prim so materials are created
            # under the actual scene root instead of at the stage root.
            root_children = list(stage.GetPseudoRoot().GetChildren())
            if root_children:
                path_prefix = str(root_children[0].GetPath())
                logger.warning(
                    f"Default prim is invalid, using root prim "
                    f"'{root_children[0].GetName()}' for material placement"
                )
            else:
                path_prefix = ""
                logger.warning(
                    "No default prim or root prims found, "
                    "creating materials at stage root"
                )
    path_prefix += "/Looks"

    UsdGeom.Scope.Define(stage, path_prefix)
    material_path = path_prefix + "/" + material_name
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")

    # Apply NodeDefAPI schema to the shader prim for proper Omniverse compatibility
    shader_prim = shader.GetPrim()
    node_def_api = UsdShade.NodeDefAPI.Apply(shader_prim)

    # Set the implementation source and MDL asset information
    node_def_api.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    node_def_api.SetSourceAsset(Sdf.AssetPath(source_asset_path), "mdl")
    node_def_api.SetSourceAssetSubIdentifier(sub_identifier, "mdl")

    # Connect shader to material outputs
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(), "out"
    )
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    # TODO: implement something that modifies the material based on hex color value
    if color is not None:
        pass

    return stage, material_path


def bind_material_to_prim(
    stage: Usd.Stage,
    material_path: str,
    prim_path: str,
    binding_strength: UsdShade.Tokens = UsdShade.Tokens.weakerThanDescendants,
) -> Usd.Stage:
    """Bind material to a prim.

    Args:
        stage: The USD stage
        material_path: Path to the material prim
        prim_path: Path to the prim to assign the material to
        binding_strength: Material binding strength (default: weakerThanDescendants)

    Returns:
        Updated stage

    Raises:
        ValueError: If the prim is an instance proxy (read-only)
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.warning(f"Prim not found at path: {prim_path}")
        return stage

    # Instance proxies are READ-ONLY in USD - cannot author properties to them
    # Skip with a warning rather than failing the entire operation
    if prim.IsInstanceProxy():
        raise ValueError(
            f"Cannot bind material to instance proxy at {prim_path}. "
            "Instance proxies are read-only. Apply materials to the prototype instead."
        )

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))

    try:
        if binding_strength == UsdShade.Tokens.weakerThanDescendants:
            try:
                import usdex.core

                if usdex.core.bindMaterial(prim, material):
                    return stage
            except Exception as usdex_error:
                logger.warning(
                    "USD-Exchange material binding failed for %s -> %s: %s",
                    prim_path,
                    material_path,
                    usdex_error,
                )

        # CRITICAL: Apply the MaterialBindingAPI schema to the prim before binding
        # This ensures the binding relationship is properly authored with the schema applied
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material, bindingStrength=binding_strength)
    except Exception as e:
        # Error: authoring to an instance proxy is not allowed
        logger.warning(f"Binding materials failed for {prim_path}: {e}")

    return stage


# Regex to strip triplanar channel suffix (_a, _b, _c) from input names
_TRIPLANAR_SUFFIX_RE = re.compile(r"_[abc]$")


def convert_custom_mdl_to_builtin(stage: Usd.Stage) -> None:
    """Replace custom MDL shader references with built-in equivalents.

    The NVCF renderer cannot load custom MDL modules. This converts:
    - CreativePBRTriplanar.mdl -> OmniPBR.mdl (with input name remapping)
    - ./Material/OmniPBR.mdl  -> OmniPBR.mdl  (fix relative path)

    Args:
        stage: USD stage to modify in-place.
    """
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue

        mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
        if not mdl_attr or not mdl_attr.IsValid():
            continue

        mdl_val = mdl_attr.Get()
        sub_attr = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
        sub_identifier = sub_attr.Get() if sub_attr and sub_attr.IsValid() else None
        if mdl_val is None or not getattr(mdl_val, "path", ""):
            if sub_identifier == "OmniPBR":
                mdl_attr.Set(Sdf.AssetPath("OmniPBR.mdl"))
            continue
        mdl_path = mdl_val.path

        # Fix local OmniPBR path -> bare name
        if mdl_path.endswith("/OmniPBR.mdl") or mdl_path.endswith("\\OmniPBR.mdl"):
            mdl_attr.Set(Sdf.AssetPath("OmniPBR.mdl"))
            continue

        # CreativePBRTriplanar -> OmniPBR
        if "CreativePBRTriplanar" not in mdl_path:
            continue

        mdl_attr.Set(Sdf.AssetPath("OmniPBR.mdl"))
        if sub_attr and sub_attr.IsValid():
            sub_attr.Set("OmniPBR")

        # Remap inputs: strip the triplanar channel suffix (_a, _b, _c)
        shader = UsdShade.Shader(prim)
        for inp in shader.GetInputs():
            old_name = inp.GetBaseName()
            new_name = _TRIPLANAR_SUFFIX_RE.sub("", old_name)
            if new_name == old_name:
                continue

            val = inp.Get()
            if val is None:
                continue
            new_inp = shader.GetInput(new_name)
            if not new_inp:
                new_inp = shader.CreateInput(new_name, inp.GetTypeName())
            new_inp.Set(val)
