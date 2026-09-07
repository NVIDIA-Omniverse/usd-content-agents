# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared inspection and representation-preserving edits for USD materials."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from world_understanding.utils.usd.asset_paths import is_uri_asset_path

from material_agent.material_library_generation.schema import TextureMapSet

from .color import srgb_to_linear

_OPENPBR_SHADER_ID = "ND_open_pbr_surface_surfaceshader"
_OMNIPBR_MDL_ASSET = "OmniPBR.mdl"
_OMNIPBR_MDL_SUBIDENTIFIER = "OmniPBR"
_OMNIPBR_TEXTURE_INPUTS = (
    "diffuse_texture",
    "normalmap_texture",
    "ORM_texture",
)
_MATERIALX_IMAGE_SHADER_IDS = frozenset(
    {
        "ND_image_color3",
        "ND_image_float",
        "ND_image_vector3",
        "ND_tiledimage_color3",
        "ND_tiledimage_float",
        "ND_tiledimage_vector3",
    }
)
_IMAGE_SHADER_IDS = _MATERIALX_IMAGE_SHADER_IDS | {"UsdUVTexture"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class MaterialGraphEditError(ValueError):
    """Raised when an existing material graph cannot be edited safely."""


@dataclass(frozen=True)
class MaterialGraphInfo:
    """Resolved source state and graph identity for one USD material."""

    usd_path: Path
    material_prim_path: str
    material_profile: str
    representation: str
    base_color: tuple[float, float, float]
    roughness: float
    metallic: float
    textures: TextureMapSet | None
    shader_ids: tuple[str, ...]
    topology_sha256: str
    opacity: float = 1.0
    transmission: float = 0.0
    ior: float = 1.5
    thin_walled: bool = False


@dataclass(frozen=True)
class MaterialGraphEdit:
    """One source graph and the values to author into its isolated copy."""

    source_usd: Path
    source_material_prim_path: str
    target_material_prim_path: str
    base_color: tuple[float, float, float] | None = None
    roughness: float | None = None
    metallic: float | None = None
    textures: TextureMapSet | None = None

    def __post_init__(self) -> None:
        controls = (self.base_color, self.roughness, self.metallic)
        if any(value is None for value in controls) and not all(
            value is None for value in controls
        ):
            raise ValueError(
                "base_color, roughness, and metallic must be provided together"
            )


@dataclass(frozen=True)
class _CapturedAsset:
    relative_prim_path: str
    attribute_name: str
    values: tuple[Sdf.AssetPath, ...]
    is_array: bool


def _shader_id(shader: UsdShade.Shader) -> str:
    value = shader.GetIdAttr().Get() if shader else None
    return str(value) if value else ""


def _mdl_source_identity(shader: UsdShade.Shader) -> str:
    if not shader:
        return ""
    try:
        source_asset = shader.GetSourceAsset("mdl")
        source_path = source_asset.path if source_asset else ""
        sub_identifier = str(shader.GetSourceAssetSubIdentifier("mdl") or "")
    except Exception:
        return ""
    if not source_path:
        return ""
    return f"mdl:{source_path}#{sub_identifier}"


def _is_omnipbr_mdl_shader(shader: UsdShade.Shader) -> bool:
    return _mdl_source_identity(shader) == (
        f"mdl:{_OMNIPBR_MDL_ASSET}#{_OMNIPBR_MDL_SUBIDENTIFIER}"
    )


def _single_connected_source(port: Any) -> Any | None:
    if not port:
        return None
    try:
        sources, invalid = port.GetConnectedSources()
    except Exception:
        return None
    if invalid or len(sources) != 1:
        return None
    return sources[0]


def _connected_shader(port: Any) -> UsdShade.Shader | None:
    source = _single_connected_source(port)
    if source is None or source.sourceType != UsdShade.AttributeType.Output:
        return None
    shader = UsdShade.Shader(source.source.GetPrim())
    return shader if shader and shader.GetPrim().IsValid() else None


def _surface_shader(
    material: UsdShade.Material, context: str
) -> UsdShade.Shader | None:
    return _connected_shader(material.GetSurfaceOutput(context))


def _detect_profile(material: UsdShade.Material) -> tuple[str, UsdShade.Shader]:
    openpbr = _surface_shader(material, "mtlx")
    if openpbr is not None and _shader_id(openpbr) == _OPENPBR_SHADER_ID:
        return "openpbr_materialx", openpbr

    mdl = _surface_shader(material, "mdl")
    if mdl is not None:
        if _is_omnipbr_mdl_shader(mdl):
            return "omnipbr_mdl", mdl
        raise MaterialGraphEditError(
            "source material uses unsupported MDL; source-backed refinement only "
            "supports the OmniPBR.mdl:OmniPBR scalar interface"
        )

    preview = _surface_shader(material, "")
    if preview is not None and _shader_id(preview) == "UsdPreviewSurface":
        return "preview_surface", preview

    raise MaterialGraphEditError(
        "source material is not a supported OpenPBR MaterialX, OmniPBR MDL, or "
        "UsdPreviewSurface graph"
    )


def _input_sources(shader_input: UsdShade.Input) -> list[Any]:
    try:
        sources, invalid = shader_input.GetConnectedSources()
    except Exception:
        return []
    return [] if invalid else list(sources)


def _find_upstream_image(shader_input: UsdShade.Input) -> UsdShade.Shader | None:
    queue = [shader_input]
    visited_inputs: set[str] = set()
    visited_shaders: set[str] = set()
    while queue:
        current = queue.pop(0)
        input_path = str(current.GetAttr().GetPath())
        if input_path in visited_inputs:
            continue
        visited_inputs.add(input_path)
        for source in _input_sources(current):
            if source.sourceType == UsdShade.AttributeType.Input:
                upstream = source.source.GetInput(source.sourceName)
                if upstream:
                    queue.append(upstream)
                continue
            if source.sourceType != UsdShade.AttributeType.Output:
                continue
            shader = UsdShade.Shader(source.source.GetPrim())
            if not shader or not shader.GetPrim().IsValid():
                continue
            shader_path = str(shader.GetPath())
            if shader_path in visited_shaders:
                continue
            visited_shaders.add(shader_path)
            if _shader_id(shader) in _IMAGE_SHADER_IDS:
                return shader
            queue.extend(
                usd_input
                for usd_input in shader.GetInputs()
                if _input_sources(usd_input)
            )
    return None


def _asset_path_from_image(
    image_shader: UsdShade.Shader | None,
    *,
    source_usd: Path,
) -> Path | None:
    if image_shader is None:
        return None
    file_input = image_shader.GetInput("file")
    value = file_input.Get() if file_input else None
    if not isinstance(value, Sdf.AssetPath) or not value.path:
        return None
    if value.resolvedPath:
        resolved = Path(value.resolvedPath).resolve()
    elif is_uri_asset_path(value.path):
        return None
    else:
        unresolved = Path(value.path).expanduser()
        resolved = (
            unresolved.resolve()
            if unresolved.is_absolute()
            else (source_usd.parent / unresolved).resolve()
        )
    return resolved if resolved.is_file() else None


def _asset_path_from_input(
    shader_input: UsdShade.Input,
    *,
    source_usd: Path,
) -> Path | None:
    queue = [shader_input]
    visited: set[str] = set()
    while queue:
        current = queue.pop(0)
        input_path = str(current.GetAttr().GetPath())
        if input_path in visited:
            continue
        visited.add(input_path)
        value = current.Get()
        if isinstance(value, Sdf.AssetPath) and value.path:
            return _resolved_asset_file(value, source_usd)
        for source in _input_sources(current):
            if source.sourceType == UsdShade.AttributeType.Input:
                upstream = source.source.GetInput(source.sourceName)
                if upstream:
                    queue.append(upstream)
    return None


def _material_input_value(
    material: UsdShade.Material,
    names: tuple[str, ...],
) -> Any | None:
    for name in names:
        material_input = material.GetInput(name)
        if material_input:
            value = material_input.Get()
            if value is not None:
                return value
    return None


def _shader_input_value(
    shader: UsdShade.Shader,
    names: tuple[str, ...],
) -> Any | None:
    for name in names:
        shader_input = shader.GetInput(name)
        if shader_input and not _input_sources(shader_input):
            value = shader_input.Get()
            if value is not None:
                return value
    return None


def _unit_scalar(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MaterialGraphEditError(
            f"source material is missing scalar {label}"
        ) from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise MaterialGraphEditError(
            f"source material {label} must be finite and in [0, 1]"
        )
    return number


def _positive_scalar(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MaterialGraphEditError(
            f"source material is missing scalar {label}"
        ) from error
    if not math.isfinite(number) or number <= 0.0:
        raise MaterialGraphEditError(
            f"source material {label} must be finite and positive"
        )
    return number


def _boolean_value(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise MaterialGraphEditError(f"source material {label} must be boolean")


def _material_or_shader_value(
    material: UsdShade.Material,
    shader: UsdShade.Shader,
    names: tuple[str, ...],
    *,
    default: Any,
) -> Any:
    value = _material_input_value(material, names)
    if value is None:
        value = _shader_input_value(shader, names)
    return default if value is None else value


def _optical_state(
    profile: str,
    material: UsdShade.Material,
    surface_shader: UsdShade.Shader,
) -> tuple[float, float, float, bool]:
    opacity_names: tuple[str, ...]
    transmission_names: tuple[str, ...]
    ior_names: tuple[str, ...]
    thin_walled_names: tuple[str, ...]
    if profile == "openpbr_materialx":
        opacity_names = ("geometry_opacity", "opacity")
        transmission_names = ("transmission_weight", "transmission")
        ior_names = ("specular_ior", "ior")
        thin_walled_names = ("geometry_thin_walled", "thin_walled")
    elif profile == "omnipbr_mdl":
        opacity_names = ("opacity", "opacity_constant")
        transmission_names = ("transmission_weight", "transmission")
        ior_names = ("ior", "ior_constant", "reflection_ior")
        thin_walled_names = ("thin_walled", "geometry_thin_walled")
    else:
        opacity_names = ("opacity",)
        transmission_names = ("transmission_weight", "transmission")
        ior_names = ("ior",)
        thin_walled_names = ("thin_walled",)

    opacity = _unit_scalar(
        _material_or_shader_value(
            material,
            surface_shader,
            opacity_names,
            default=1.0,
        ),
        label="opacity",
    )
    transmission = _unit_scalar(
        _material_or_shader_value(
            material,
            surface_shader,
            transmission_names,
            default=0.0,
        ),
        label="transmission",
    )
    ior = _positive_scalar(
        _material_or_shader_value(
            material,
            surface_shader,
            ior_names,
            default=1.5,
        ),
        label="ior",
    )
    thin_walled = _boolean_value(
        _material_or_shader_value(
            material,
            surface_shader,
            thin_walled_names,
            default=False,
        ),
        label="thin_walled",
    )
    return opacity, transmission, ior, thin_walled


def _unit_color(value: Any) -> tuple[float, float, float]:
    try:
        color = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError) as error:
        raise MaterialGraphEditError("source material is missing base color") from error
    if not all(math.isfinite(channel) and 0.0 <= channel <= 1.0 for channel in color):
        raise MaterialGraphEditError(
            "source material base color channels must be finite and in [0, 1]"
        )
    return color


def _mean_texture_features(
    textures: TextureMapSet,
) -> tuple[tuple[float, float, float], float, float]:
    rgba = _load_8bit_texture(textures.albedo, mode="RGBA", channel="albedo")
    orm = _load_8bit_texture(textures.orm, mode="RGB", channel="orm")
    albedo = srgb_to_linear(rgba[..., :3]).reshape(-1, 3)
    alpha = rgba[..., 3].reshape(-1)
    weight_sum = float(alpha.sum())
    if weight_sum <= 0.0:
        raise MaterialGraphEditError("source albedo has no visible pixels")
    color_values = (albedo * alpha[:, None]).sum(axis=0) / weight_sum
    orm_values = orm.reshape(-1, 3).mean(axis=0)
    return (
        (float(color_values[0]), float(color_values[1]), float(color_values[2])),
        float(orm_values[1]),
        float(orm_values[2]),
    )


def _load_8bit_texture(
    path: Path,
    *,
    mode: str,
    channel: str,
) -> np.ndarray:
    """Load an 8-bit source map without silently clipping higher bit depths."""

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode == "F" or image.mode == "I" or image.mode.startswith("I;16"):
                raise MaterialGraphEditError(
                    f"source {channel} texture must use 8-bit channels"
                )
            return np.asarray(image.convert(mode), dtype=np.float64) / 255.0
    except MaterialGraphEditError:
        raise
    except OSError as error:
        raise MaterialGraphEditError(
            f"source {channel} texture is not a decodable image: {path}"
        ) from error


def _texture_inputs(
    profile: str,
    surface_shader: UsdShade.Shader,
) -> tuple[UsdShade.Input, UsdShade.Input, UsdShade.Input, UsdShade.Input]:
    if profile == "openpbr_materialx":
        return (
            surface_shader.GetInput("base_color"),
            surface_shader.GetInput("geometry_normal"),
            surface_shader.GetInput("specular_roughness"),
            surface_shader.GetInput("base_metalness"),
        )
    if profile == "omnipbr_mdl":
        raise MaterialGraphEditError(
            "texture-backed OmniPBR MDL refinement is not supported"
        )
    return (
        surface_shader.GetInput("diffuseColor"),
        surface_shader.GetInput("normal"),
        surface_shader.GetInput("roughness"),
        surface_shader.GetInput("metallic"),
    )


def _discover_textures(
    profile: str,
    surface_shader: UsdShade.Shader,
    *,
    source_usd: Path,
    allow_textured_omnipbr: bool,
) -> TextureMapSet | None:
    if profile == "omnipbr_mdl":
        texture_inputs = tuple(
            surface_shader.GetInput(name) for name in _OMNIPBR_TEXTURE_INPUTS
        )
        authored_inputs = tuple(
            shader_input
            for shader_input in texture_inputs
            if shader_input
            and (
                (
                    shader_input.Get().path
                    if isinstance(shader_input.Get(), Sdf.AssetPath)
                    else str(shader_input.Get() or "")
                )
                or _input_sources(shader_input)
            )
        )
        if not authored_inputs:
            return None
        if not allow_textured_omnipbr:
            raise MaterialGraphEditError(
                "texture-backed OmniPBR MDL refinement is not supported"
            )
        paths = tuple(
            _asset_path_from_input(shader_input, source_usd=source_usd)
            for shader_input in texture_inputs
            if shader_input
        )
        if len(paths) != len(_OMNIPBR_TEXTURE_INPUTS) or any(
            path is None for path in paths
        ):
            raise MaterialGraphEditError(
                "textured OmniPBR material must expose complete diffuse, normal, "
                "and ORM asset inputs"
            )
        return TextureMapSet(albedo=paths[0], normal=paths[1], orm=paths[2])

    color_input, normal_input, roughness_input, metallic_input = _texture_inputs(
        profile, surface_shader
    )
    paths = tuple(
        _asset_path_from_image(
            _find_upstream_image(shader_input), source_usd=source_usd
        )
        if shader_input
        else None
        for shader_input in (
            color_input,
            normal_input,
            roughness_input,
            metallic_input,
        )
    )
    if not any(paths):
        return None
    albedo, normal, roughness, metallic = paths
    if (
        albedo is None
        or normal is None
        or roughness is None
        or metallic is None
        or roughness != metallic
    ):
        raise MaterialGraphEditError(
            "textured source material must expose a complete albedo, normal, and "
            "shared ORM graph for representation-preserving refinement"
        )
    return TextureMapSet(albedo=albedo, normal=normal, orm=roughness)


def _normalized_property_path(path: str, material_path: str) -> str:
    return path.replace(material_path, "@MATERIAL@", 1)


def _topology_sha256(material_prim: Usd.Prim) -> str:
    material_path = str(material_prim.GetPath())
    topology: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(material_prim):
        entry: dict[str, Any] = {
            "path": _normalized_property_path(str(prim.GetPath()), material_path),
            "type": prim.GetTypeName(),
        }
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            shader_id = _shader_id(shader)
            mdl_identity = _mdl_source_identity(shader)
            if shader_id:
                entry["shader_id"] = shader_id
            if mdl_identity:
                entry["mdl_source"] = mdl_identity
        connections: dict[str, list[str]] = {}
        for attribute in prim.GetAttributes():
            authored = [
                _normalized_property_path(str(path), material_path)
                for path in attribute.GetConnections()
            ]
            if authored:
                connections[attribute.GetName()] = sorted(authored)
        if connections:
            entry["connections"] = connections
        topology.append(entry)
    payload = json.dumps(topology, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inspect_material_graph(
    usd_path: str | Path,
    material_prim_path: str,
    *,
    allow_textured_omnipbr: bool = False,
) -> MaterialGraphInfo:
    """Inspect one supported material without changing its source stage."""

    source_usd = Path(usd_path).expanduser().resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(f"source USD does not exist: {source_usd}")
    if not Sdf.Path.IsValidPathString(material_prim_path):
        raise MaterialGraphEditError("material_prim_path is not a valid USD path")
    path = Sdf.Path(material_prim_path)
    if not path.IsAbsolutePath() or path.IsPropertyPath():
        raise MaterialGraphEditError("material_prim_path must be an absolute prim path")

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise MaterialGraphEditError(f"failed to open source USD: {source_usd}")
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdShade.Material):
        raise MaterialGraphEditError(
            f"material_prim_path does not identify a UsdShade.Material: {path}"
        )
    if prim.IsInstanceProxy():
        raise MaterialGraphEditError(
            f"material_prim_path identifies a read-only instance proxy: {path}"
        )
    if prim.IsInstance() or prim.IsInstanceable():
        stage.SetEditTarget(stage.GetSessionLayer())
        prim.SetInstanceable(False)
    material = UsdShade.Material(prim)
    profile, surface_shader = _detect_profile(material)
    textures = _discover_textures(
        profile,
        surface_shader,
        source_usd=source_usd,
        allow_textured_omnipbr=allow_textured_omnipbr,
    )
    opacity, transmission, ior, thin_walled = _optical_state(
        profile,
        material,
        surface_shader,
    )

    if textures is not None:
        base_color, roughness, metallic = _mean_texture_features(textures)
        representation = "textured_pbr"
    else:
        if profile == "openpbr_materialx":
            base_color_value = _material_input_value(material, ("base_color",))
            roughness_value = _material_input_value(
                material, ("specular_roughness", "roughness")
            )
            metallic_value = _material_input_value(
                material, ("base_metalness", "metallic")
            )
            if base_color_value is None:
                base_color_value = _shader_input_value(surface_shader, ("base_color",))
            roughness_value = (
                roughness_value
                if roughness_value is not None
                else _shader_input_value(
                    surface_shader, ("specular_roughness", "roughness")
                )
            )
            metallic_value = (
                metallic_value
                if metallic_value is not None
                else _shader_input_value(surface_shader, ("base_metalness", "metallic"))
            )
        elif profile == "omnipbr_mdl":
            base_color_value = _material_input_value(
                material,
                ("diffuseColor", "diffuse_color_constant", "diffuse_color"),
            )
            roughness_value = _material_input_value(
                material,
                ("roughness", "reflection_roughness_constant", "roughness_constant"),
            )
            metallic_value = _material_input_value(
                material,
                ("metallic", "metallic_constant", "metalness_constant"),
            )
            if base_color_value is None:
                base_color_value = _shader_input_value(
                    surface_shader,
                    ("diffuse_color_constant", "diffuse_tint", "diffuse_color"),
                )
            if roughness_value is None:
                roughness_value = _shader_input_value(
                    surface_shader,
                    ("reflection_roughness_constant", "roughness_constant"),
                )
            if metallic_value is None:
                metallic_value = _shader_input_value(
                    surface_shader,
                    ("metallic_constant", "metalness_constant", "metallic"),
                )
        else:
            base_color_value = _shader_input_value(surface_shader, ("diffuseColor",))
            roughness_value = _shader_input_value(surface_shader, ("roughness",))
            metallic_value = _shader_input_value(surface_shader, ("metallic",))
        base_color = _unit_color(base_color_value)
        roughness = _unit_scalar(roughness_value, label="roughness")
        metallic = _unit_scalar(metallic_value, label="metallic")
        representation = "scalar_pbr"

    shader_ids = tuple(
        sorted(
            _shader_id(UsdShade.Shader(descendant))
            or _mdl_source_identity(UsdShade.Shader(descendant))
            for descendant in Usd.PrimRange(prim)
            if descendant.IsA(UsdShade.Shader)
            and (
                _shader_id(UsdShade.Shader(descendant))
                or _mdl_source_identity(UsdShade.Shader(descendant))
            )
        )
    )
    return MaterialGraphInfo(
        usd_path=source_usd,
        material_prim_path=str(path),
        material_profile=profile,
        representation=representation,
        base_color=base_color,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
        transmission=transmission,
        ior=ior,
        thin_walled=thin_walled,
        textures=textures,
        shader_ids=shader_ids,
        topology_sha256=_topology_sha256(prim),
    )


def _ensure_parent_scopes(stage: Usd.Stage, path: Sdf.Path) -> None:
    parents: list[Sdf.Path] = []
    parent = path.GetParentPath()
    while parent != Sdf.Path.absoluteRootPath:
        parents.append(parent)
        parent = parent.GetParentPath()
    for parent_path in reversed(parents):
        if not stage.GetPrimAtPath(parent_path):
            UsdGeom.Scope.Define(stage, parent_path)


def _relative_prim_path(prim_path: Sdf.Path, root_path: Sdf.Path) -> str:
    if prim_path == root_path:
        return ""
    return str(prim_path).removeprefix(f"{root_path}/")


def _capture_assets(
    stage: Usd.Stage, material_path: Sdf.Path
) -> tuple[_CapturedAsset, ...]:
    captures: list[_CapturedAsset] = []
    material_prim = stage.GetPrimAtPath(material_path)
    for prim in Usd.PrimRange(material_prim):
        relative_prim = _relative_prim_path(prim.GetPath(), material_path)
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath):
                captures.append(
                    _CapturedAsset(
                        relative_prim_path=relative_prim,
                        attribute_name=attribute.GetName(),
                        values=(value,),
                        is_array=False,
                    )
                )
            elif isinstance(value, Sdf.AssetPathArray):
                captures.append(
                    _CapturedAsset(
                        relative_prim_path=relative_prim,
                        attribute_name=attribute.GetName(),
                        values=tuple(value),
                        is_array=True,
                    )
                )
    return tuple(captures)


def _resolved_asset_file(asset: Sdf.AssetPath, source_usd: Path) -> Path | None:
    if not asset.path or is_uri_asset_path(asset.path):
        return None
    if asset.resolvedPath:
        candidate = Path(asset.resolvedPath).resolve()
    else:
        unresolved = Path(asset.path).expanduser()
        candidate = (
            unresolved.resolve()
            if unresolved.is_absolute()
            else (source_usd.parent / unresolved).resolve()
        )
    return candidate if candidate.is_file() else None


def _package_asset(
    source: Path,
    *,
    output_usd: Path,
    asset_dir: Path,
) -> Sdf.AssetPath:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    destination = asset_dir / f"{digest}_{source.name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    relative = os.path.relpath(destination, output_usd.parent).replace("\\", "/")
    return Sdf.AssetPath(relative)


def _restore_captured_assets(
    stage: Usd.Stage,
    captures: tuple[_CapturedAsset, ...],
    *,
    source_usd: Path,
    target_material_path: Sdf.Path,
    output_usd: Path,
) -> None:
    safe_name = _SAFE_NAME_RE.sub("_", target_material_path.name) or "material"
    asset_dir = output_usd.parent / "source_assets" / safe_name
    for capture in captures:
        prim_path = target_material_path
        if capture.relative_prim_path:
            prim_path = prim_path.AppendPath(capture.relative_prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise MaterialGraphEditError(
                f"copied material graph is missing expected prim: {prim_path}"
            )
        attribute = prim.GetAttribute(capture.attribute_name)
        if not attribute:
            continue
        packaged: list[Sdf.AssetPath] = []
        for asset in capture.values:
            source_file = _resolved_asset_file(asset, source_usd)
            packaged.append(
                _package_asset(
                    source_file,
                    output_usd=output_usd,
                    asset_dir=asset_dir,
                )
                if source_file is not None
                else asset
            )
        if capture.is_array:
            attribute.Set(Sdf.AssetPathArray(packaged))
        elif packaged:
            attribute.Set(packaged[0])


def _assert_connections_in_scope(material_prim: Usd.Prim) -> None:
    """Reject copied graphs whose connections leave the material namespace."""

    material_path = material_prim.GetPath()
    for prim in Usd.PrimRange(material_prim):
        for attribute in prim.GetAttributes():
            for target in attribute.GetConnections():
                if not target.GetPrimPath().HasPrefix(material_path):
                    raise MaterialGraphEditError(
                        "material graph connects to a prim outside the material "
                        f"scope and cannot be published portably: {target}"
                    )


def _set_material_input(
    material: UsdShade.Material,
    names: tuple[str, ...],
    value: Any,
) -> bool:
    updated = False
    for name in names:
        material_input = material.GetInput(name)
        if material_input:
            material_input.Set(value)
            updated = True
    return updated


def _set_unconnected_shader_input(
    shader: UsdShade.Shader,
    names: tuple[str, ...],
    value: Any,
) -> bool:
    updated = False
    for name in names:
        shader_input = shader.GetInput(name)
        if shader_input and not _input_sources(shader_input):
            shader_input.Set(value)
            updated = True
    return updated


def _sync_openpbr_preview_fallback(
    material: UsdShade.Material,
    *,
    base_color: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> None:
    preview = _surface_shader(material, "")
    if preview is None or _shader_id(preview) != "UsdPreviewSurface":
        return
    _set_unconnected_shader_input(preview, ("diffuseColor",), Gf.Vec3f(*base_color))
    _set_unconnected_shader_input(preview, ("roughness",), roughness)
    _set_unconnected_shader_input(preview, ("metallic",), metallic)


def _edit_scalar_controls(
    material: UsdShade.Material,
    profile: str,
    surface_shader: UsdShade.Shader,
    *,
    base_color: tuple[float, float, float],
    roughness: float,
    metallic: float,
    require_all: bool,
) -> None:
    color = Gf.Vec3f(*base_color)
    if profile == "openpbr_materialx":
        updates = {
            "base_color": _set_material_input(material, ("base_color",), color),
            "roughness": _set_material_input(
                material, ("specular_roughness", "roughness"), roughness
            ),
            "metallic": _set_material_input(
                material, ("base_metalness", "metallic"), metallic
            ),
        }
        updates["base_color"] = (
            _set_unconnected_shader_input(surface_shader, ("base_color",), color)
            or updates["base_color"]
        )
        updates["roughness"] = (
            _set_unconnected_shader_input(
                surface_shader, ("specular_roughness", "roughness"), roughness
            )
            or updates["roughness"]
        )
        updates["metallic"] = (
            _set_unconnected_shader_input(
                surface_shader, ("base_metalness", "metallic"), metallic
            )
            or updates["metallic"]
        )
        _sync_openpbr_preview_fallback(
            material,
            base_color=base_color,
            roughness=roughness,
            metallic=metallic,
        )
    elif profile == "omnipbr_mdl":
        updates = {
            "base_color": _set_material_input(
                material,
                ("diffuseColor", "diffuse_color_constant", "diffuse_color"),
                color,
            ),
            "roughness": _set_material_input(
                material,
                ("roughness", "reflection_roughness_constant", "roughness_constant"),
                roughness,
            ),
            "metallic": _set_material_input(
                material,
                ("metallic", "metallic_constant", "metalness_constant"),
                metallic,
            ),
        }
        updates["base_color"] = (
            _set_unconnected_shader_input(
                surface_shader,
                ("diffuse_color_constant", "diffuse_tint", "diffuse_color"),
                color,
            )
            or updates["base_color"]
        )
        updates["roughness"] = (
            _set_unconnected_shader_input(
                surface_shader,
                ("reflection_roughness_constant", "roughness_constant"),
                roughness,
            )
            or updates["roughness"]
        )
        updates["metallic"] = (
            _set_unconnected_shader_input(
                surface_shader,
                ("metallic_constant", "metalness_constant", "metallic"),
                metallic,
            )
            or updates["metallic"]
        )
    else:
        updates = {
            "base_color": _set_unconnected_shader_input(
                surface_shader, ("diffuseColor",), color
            ),
            "roughness": _set_unconnected_shader_input(
                surface_shader, ("roughness",), roughness
            ),
            "metallic": _set_unconnected_shader_input(
                surface_shader, ("metallic",), metallic
            ),
        }
    missing = sorted(name for name, updated in updates.items() if not updated)
    if missing and require_all:
        raise MaterialGraphEditError(
            "source material does not expose editable scalar control(s): "
            + ", ".join(missing)
        )


def _pack_replacement_texture(
    source: Path,
    *,
    channel: str,
    target_material_path: Sdf.Path,
    output_usd: Path,
) -> Sdf.AssetPath:
    safe_name = _SAFE_NAME_RE.sub("_", target_material_path.name) or "material"
    suffix = source.suffix.lower() or ".png"
    destination = output_usd.parent / "textures" / safe_name / f"{channel}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    relative = os.path.relpath(destination, output_usd.parent).replace("\\", "/")
    return Sdf.AssetPath(relative)


def _replace_image_file(
    shader_input: UsdShade.Input,
    asset: Sdf.AssetPath,
    *,
    color_space: str,
    channel: str,
) -> None:
    image = _find_upstream_image(shader_input)
    if image is None:
        raise MaterialGraphEditError(
            f"textured source graph has no editable {channel} image node"
        )
    file_input = image.GetInput("file")
    if not file_input:
        raise MaterialGraphEditError(
            f"textured source graph {channel} image node has no file input"
        )
    file_input.Set(asset)
    file_input.GetAttr().SetColorSpace(color_space)


def _edit_texture_paths(
    profile: str,
    surface_shader: UsdShade.Shader,
    textures: TextureMapSet,
    *,
    target_material_path: Sdf.Path,
    output_usd: Path,
) -> None:
    color_input, normal_input, roughness_input, metallic_input = _texture_inputs(
        profile, surface_shader
    )
    if not all((color_input, normal_input, roughness_input, metallic_input)):
        raise MaterialGraphEditError(
            "textured source graph is missing required PBR shader inputs"
        )
    albedo = _pack_replacement_texture(
        textures.albedo,
        channel="albedo",
        target_material_path=target_material_path,
        output_usd=output_usd,
    )
    normal = _pack_replacement_texture(
        textures.normal,
        channel="normal",
        target_material_path=target_material_path,
        output_usd=output_usd,
    )
    orm = _pack_replacement_texture(
        textures.orm,
        channel="orm",
        target_material_path=target_material_path,
        output_usd=output_usd,
    )
    _replace_image_file(color_input, albedo, color_space="sRGB", channel="albedo")
    _replace_image_file(normal_input, normal, color_space="raw", channel="normal")
    _replace_image_file(roughness_input, orm, color_space="raw", channel="roughness")
    _replace_image_file(metallic_input, orm, color_space="raw", channel="metallic")


def write_edited_material_graphs(
    output_usd_path: str | Path,
    edits: tuple[MaterialGraphEdit, ...] | list[MaterialGraphEdit],
) -> tuple[MaterialGraphInfo, ...]:
    """Clone source graphs into one library and author only requested changes."""

    if not edits:
        raise ValueError("at least one source material graph edit is required")
    output_usd = Path(output_usd_path).expanduser().resolve()
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    if output_usd.exists():
        output_usd.unlink()
    output_stage = Usd.Stage.CreateNew(str(output_usd))
    if output_stage is None:
        raise MaterialGraphEditError(f"failed to create output USD: {output_usd}")
    output_stage.GetRootLayer().Save()

    planned: list[tuple[MaterialGraphInfo, MaterialGraphEdit, Sdf.Path]] = []
    seen_targets: set[str] = set()
    source_material_paths: dict[Path, set[str]] = {}
    for edit in edits:
        source_info = inspect_material_graph(
            edit.source_usd, edit.source_material_prim_path
        )
        target_path = Sdf.Path(edit.target_material_prim_path)
        if not target_path.IsAbsolutePath() or target_path.IsPropertyPath():
            raise MaterialGraphEditError(
                "target material path must be an absolute USD prim path"
            )
        if str(target_path) in seen_targets:
            raise MaterialGraphEditError(
                f"duplicate target material path: {target_path}"
            )
        seen_targets.add(str(target_path))
        planned.append((source_info, edit, target_path))
        source_material_paths.setdefault(source_info.usd_path, set()).add(
            source_info.material_prim_path
        )

    flattened_sources: dict[Path, Sdf.Layer] = {}
    captured_assets: dict[tuple[Path, str], tuple[_CapturedAsset, ...]] = {}
    for source_usd, material_paths in source_material_paths.items():
        source_stage = Usd.Stage.Open(str(source_usd))
        if source_stage is None:
            raise MaterialGraphEditError(f"failed to reopen source USD: {source_usd}")
        session_target_selected = False
        for material_path in sorted(material_paths):
            source_prim = source_stage.GetPrimAtPath(Sdf.Path(material_path))
            if source_prim.IsInstance() or source_prim.IsInstanceable():
                if not session_target_selected:
                    source_stage.SetEditTarget(source_stage.GetSessionLayer())
                    session_target_selected = True
                source_prim.SetInstanceable(False)
        for material_path in sorted(material_paths):
            captured_assets[(source_usd, material_path)] = _capture_assets(
                source_stage,
                Sdf.Path(material_path),
            )
        flattened_sources[source_usd] = source_stage.Flatten()

    expected: list[tuple[MaterialGraphInfo, MaterialGraphEdit]] = []
    for source_info, edit, target_path in planned:
        source_path = Sdf.Path(source_info.material_prim_path)
        captures = captured_assets[
            (source_info.usd_path, source_info.material_prim_path)
        ]
        flattened = flattened_sources[source_info.usd_path]
        output_stage = Usd.Stage.Open(str(output_usd))
        if output_stage is None:
            raise MaterialGraphEditError(f"failed to reopen output USD: {output_usd}")
        _ensure_parent_scopes(output_stage, target_path)
        output_stage.GetRootLayer().Save()
        if not Sdf.CopySpec(
            flattened,
            source_path,
            output_stage.GetRootLayer(),
            target_path,
        ):
            raise MaterialGraphEditError(
                f"failed to copy material graph {source_path} to {target_path}"
            )
        output_stage.GetRootLayer().Save()
        output_stage = Usd.Stage.Open(str(output_usd))
        if output_stage is None:
            raise MaterialGraphEditError(f"failed to reopen output USD: {output_usd}")
        _restore_captured_assets(
            output_stage,
            captures,
            source_usd=source_info.usd_path,
            target_material_path=target_path,
            output_usd=output_usd,
        )
        target_prim = output_stage.GetPrimAtPath(target_path)
        _assert_connections_in_scope(target_prim)
        material = UsdShade.Material(target_prim)
        profile, surface_shader = _detect_profile(material)
        if profile != source_info.material_profile:
            raise MaterialGraphEditError(
                "copied material graph changed profile during composition"
            )
        if edit.base_color is not None:
            assert edit.roughness is not None
            assert edit.metallic is not None
            _edit_scalar_controls(
                material,
                profile,
                surface_shader,
                base_color=edit.base_color,
                roughness=edit.roughness,
                metallic=edit.metallic,
                require_all=source_info.textures is None,
            )
        if edit.textures is not None:
            _edit_texture_paths(
                profile,
                surface_shader,
                edit.textures,
                target_material_path=target_path,
                output_usd=output_usd,
            )
        output_stage.GetRootLayer().Save()
        if _topology_sha256(target_prim) != source_info.topology_sha256:
            raise MaterialGraphEditError(
                "material graph topology changed while authoring candidate values"
            )
        expected.append((source_info, edit))

    results: list[MaterialGraphInfo] = []
    for source_info, edit in expected:
        result = inspect_material_graph(output_usd, edit.target_material_prim_path)
        if result.material_profile != source_info.material_profile:
            raise MaterialGraphEditError("published material profile changed")
        if result.representation != source_info.representation:
            raise MaterialGraphEditError("published material representation changed")
        if result.topology_sha256 != source_info.topology_sha256:
            raise MaterialGraphEditError("published material topology changed")
        results.append(result)
    return tuple(results)


__all__ = [
    "MaterialGraphEdit",
    "MaterialGraphEditError",
    "MaterialGraphInfo",
    "inspect_material_graph",
    "write_edited_material_graphs",
]
