# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD material introspection for OpenPBR, MaterialX, and MDL materials.

Discovers materials in a USD stage, extracts direct OpenPBR attributes and
shader-network properties, and identifies which geometry prims are bound to
each material.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdShade

from texture_agent.functions.detail_policy import (
    apply_detail_policy_to_prompt,
    normalize_detail_policy,
)

logger = logging.getLogger(__name__)


@dataclass
class MaterialInfo:
    """Information about a discovered material in a USD stage."""

    prim_path: str
    """Prim path of the material (e.g., '/World/Looks/Steel_Carbon')."""

    name: str
    """Material prim name (e.g., 'Steel_Carbon')."""

    bound_prim_paths: list[str] = field(default_factory=list)
    """Geometry prim paths bound to this material."""

    base_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Constant base_color value (linear sRGB, 0-1)."""

    base_color_texture: str | None = None
    """Existing albedo/base color texture path, or None if empty."""

    base_metalness: float | None = None
    """Constant base_metalness value."""

    specular_roughness: float | None = None
    """Constant specular_roughness value."""

    has_existing_texture: bool = False
    """True if the material has any authored texture input."""

    bound_subset_paths: list[str] = field(default_factory=list)
    """Material-binding GeomSubset paths bound to this material."""

    material_alias_paths: list[str] = field(default_factory=list)
    """Composed paths that resolve to the same effective material identity."""


@dataclass(frozen=True)
class MaterialDiscoverySkip:
    """An authored material excluded from effective-bound discovery."""

    material_prim_path: str
    material_name: str
    reason_code: str
    reason: str


@dataclass(frozen=True)
class EffectiveMaterialDiscovery:
    """Deterministic authored and effective material discovery result.

    ``authored_materials`` includes composed material definitions even when they
    are unbound. ``effective_materials`` contains only materials that resolve on
    an in-scope renderable Gprim or material-binding GeomSubset. Instance proxy
    bindings are reduced by shared prototype identity so repeated instances do
    not multiply candidates; the first composed material path is reported
    rather than USD's internal prototype path.
    """

    authored_materials: tuple[MaterialInfo, ...]
    effective_materials: tuple[MaterialInfo, ...]
    renderable_prim_paths: tuple[str, ...]
    renderable_subset_paths: tuple[str, ...]
    skipped_materials: tuple[MaterialDiscoverySkip, ...]

    @property
    def authored_material_count(self) -> int:
        return len(self.authored_materials)

    @property
    def renderable_prim_count(self) -> int:
        return len(self.renderable_prim_paths)

    @property
    def renderable_subset_count(self) -> int:
        return len(self.renderable_subset_paths)

    @property
    def effective_bound_material_count(self) -> int:
        return len(self.effective_materials)


@dataclass
class _MaterialBindingMembers:
    """Mutable accumulator used while traversing effective bindings once."""

    material_prim: Usd.Prim
    material_alias_paths: set[str] = field(default_factory=set)
    prim_paths: set[str] = field(default_factory=set)
    subset_paths: set[str] = field(default_factory=set)


@dataclass
class _MaterialCandidate:
    """One canonical material definition and its composed path aliases."""

    material_prim: Usd.Prim
    alias_paths: set[str] = field(default_factory=set)


_ALBEDO_TEXTURE_INPUTS = {
    "diffuse_texture",
    "diffusecolor_texture",
    "diffuse_color_texture",
    "albedo_texture",
    "basecolor_texture",
    "base_color_texture",
    "base_color_texture_file",
}

_TEXTURE_INPUTS = _ALBEDO_TEXTURE_INPUTS | {
    "normalmap_texture",
    "normal_texture",
    "normal_map_texture",
    "orm_texture",
    "reflectionroughness_texture",
    "roughness_texture",
    "specular_roughness_texture",
    "specular_roughness_texture_file",
    "metallic_texture",
    "metalness_texture",
    "base_metalness_texture_file",
    "geometry_normal_texture_file",
    "coat_normal_texture_file",
    "geometry_opacity_texture_file",
}

_TEXTURE_READER_FILE_INPUTS = {"file", "filename"}

_TEXTURE_READER_ID_TOKENS = ("texture", "image", "texcoord")

_ALBEDO_NAME_TOKENS = (
    "albedo",
    "basecolor",
    "base_color",
    "diffuse",
    "diffusecolor",
    "diffuse_color",
)

_BASE_COLOR_INPUTS = (
    "base_color",
    "diffuse_tint",
    "diffuse_color",
    "diffuseColor",
    "albedo",
)

_METALNESS_INPUTS = ("base_metalness", "metalness", "metallic")

_ROUGHNESS_INPUTS = (
    "specular_roughness",
    "roughness",
    "reflectionroughness",
)


def _read_color3f(prim: Usd.Prim, attr_name: str) -> tuple[float, float, float] | None:
    """Read a color3f attribute from a prim."""
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    val = attr.Get()
    if val is None:
        return None
    return (float(val[0]), float(val[1]), float(val[2]))


def _coerce_color3f(value: object) -> tuple[float, float, float] | None:
    """Coerce a USD color/vector value into a plain RGB tuple."""
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    """Coerce a USD scalar value into a float."""
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_float(prim: Usd.Prim, attr_name: str) -> float | None:
    """Read a float attribute from a prim."""
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    val = attr.Get()
    return _coerce_float(val)


def _read_asset_path(prim: Usd.Prim, attr_name: str) -> str | None:
    """Read an asset path attribute, returning None if empty or '@@'."""
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    val = attr.Get()
    if val is None:
        return None
    path = val.path if hasattr(val, "path") else str(val)
    if not path or path == "@@":
        return None
    return path


def _coerce_texture_path(value: object) -> str | None:
    """Return a normalized path string for authored asset/string texture inputs."""
    if value is None:
        return None
    if hasattr(value, "path"):
        path = str(value.path)
    elif isinstance(value, str):
        path = value
    else:
        return None
    if not path or path == "@@":
        return None
    return path


def _iter_shader_prims(prim: Usd.Prim) -> Iterator[Usd.Prim]:
    """Yield shader descendants under a material prim."""
    for child in prim.GetAllChildren():
        if child.IsA(UsdShade.Shader):
            yield child
        yield from _iter_shader_prims(child)


def _shader_id(shader: UsdShade.Shader) -> str:
    """Return the shader id token, if authored."""
    shader_id = shader.GetIdAttr().Get()
    return str(shader_id).lower() if shader_id is not None else ""


def _compact_token(value: str) -> str:
    """Normalize names for fuzzy USD shader/input matching."""
    return value.lower().replace("_", "").replace("-", "")


def _is_texture_reader_file_input(
    input_name: str,
    shader_name: str,
    shader_id: str,
) -> bool:
    """Return True for MaterialX/UsdUVTexture-style file inputs."""
    if input_name not in _TEXTURE_READER_FILE_INPUTS:
        return False
    shader_key = _compact_token(f"{shader_name} {shader_id}")
    return any(
        _compact_token(token) in shader_key
        for token in (*_TEXTURE_READER_ID_TOKENS, *_ALBEDO_NAME_TOKENS)
    )


def _is_albedo_texture_name(name: str) -> bool:
    """Return True if a shader or input name describes an albedo texture."""
    name_key = _compact_token(name)
    return any(_compact_token(token) in name_key for token in _ALBEDO_NAME_TOKENS)


def _read_shader_color(prim: Usd.Prim) -> tuple[float, float, float] | None:
    """Read common shader-network base color inputs."""
    for shader_prim in _iter_shader_prims(prim):
        shader = UsdShade.Shader(shader_prim)
        for input_name in _BASE_COLOR_INPUTS:
            shader_input = shader.GetInput(input_name)
            if not shader_input:
                continue
            color = _coerce_color3f(shader_input.Get())
            if color is not None:
                return color
    return None


def _read_shader_float(prim: Usd.Prim, input_names: tuple[str, ...]) -> float | None:
    """Read common shader-network float inputs."""
    for shader_prim in _iter_shader_prims(prim):
        shader = UsdShade.Shader(shader_prim)
        for input_name in input_names:
            shader_input = shader.GetInput(input_name)
            if not shader_input:
                continue
            val = _coerce_float(shader_input.Get())
            if val is not None:
                return val
    return None


def _find_existing_texture_paths(prim: Usd.Prim) -> tuple[str | None, bool]:
    """Find authored texture inputs on OpenPBR, MaterialX, and MDL materials."""
    base_color_texture: str | None = None
    has_texture = False

    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        base_name = attr_name.rsplit(":", 1)[-1].lower()
        if "texture" not in base_name:
            continue
        path = _coerce_texture_path(attr.Get())
        if path is None:
            continue
        has_texture = True
        if base_name in _ALBEDO_TEXTURE_INPUTS and base_color_texture is None:
            base_color_texture = path

    for shader_prim in _iter_shader_prims(prim):
        shader = UsdShade.Shader(shader_prim)
        shader_name = shader_prim.GetName().lower()
        shader_id = _shader_id(shader)
        for shader_input in shader.GetInputs():
            base_name = shader_input.GetBaseName()
            normalized = base_name.lower()
            is_texture_reader_file = _is_texture_reader_file_input(
                normalized,
                shader_name,
                shader_id,
            )
            if (
                normalized not in _TEXTURE_INPUTS
                and not normalized.endswith("_texture")
                and not normalized.endswith("_texture_file")
                and not is_texture_reader_file
            ):
                continue
            path = _coerce_texture_path(shader_input.Get())
            if path is None:
                continue
            has_texture = True
            if (
                normalized in _ALBEDO_TEXTURE_INPUTS
                or (is_texture_reader_file and _is_albedo_texture_name(shader_name))
            ) and base_color_texture is None:
                base_color_texture = path

    return base_color_texture, has_texture


def _find_bound_prims(stage: Usd.Stage, material_path: str) -> list[str]:
    """Find all geometry prims bound to a given material."""
    return _build_material_bound_prim_index(stage).get(material_path, [])


def _iter_renderable_prims(stage: Usd.Stage) -> Iterator[Usd.Prim]:
    """Yield default-predicate prims, including descendants of instances."""
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    yield from Usd.PrimRange.Stage(stage, predicate)


def _canonical_material_prim(material_prim: Usd.Prim) -> Usd.Prim:
    """Return the shared prototype prim for an instance-proxy material."""
    if material_prim.IsInstanceProxy():
        prototype_prim = material_prim.GetPrimInPrototype()
        if prototype_prim and prototype_prim.IsValid():
            return prototype_prim
    return material_prim


def _add_material_binding(
    bindings: dict[str, _MaterialBindingMembers],
    material: UsdShade.Material,
    *,
    prim_path: str | None = None,
    subset_path: str | None = None,
) -> None:
    """Add one resolved binding to its canonical material accumulator."""
    if not material:
        return
    composed_prim = material.GetPrim()
    if not composed_prim or not composed_prim.IsValid():
        return
    canonical_prim = _canonical_material_prim(composed_prim)
    canonical_path = str(canonical_prim.GetPath())
    members = bindings.setdefault(
        canonical_path,
        _MaterialBindingMembers(material_prim=canonical_prim),
    )
    members.material_alias_paths.add(str(composed_prim.GetPath()))
    if prim_path is not None:
        members.prim_paths.add(prim_path)
    if subset_path is not None:
        members.subset_paths.add(subset_path)


def _add_material_alias(
    bindings: dict[str, _MaterialBindingMembers],
    material: UsdShade.Material,
    alias_material: UsdShade.Material,
) -> None:
    """Record that ``alias_material`` should receive the target material output."""
    if not material or not alias_material:
        return
    composed_prim = material.GetPrim()
    alias_prim = alias_material.GetPrim()
    if (
        not composed_prim
        or not composed_prim.IsValid()
        or not alias_prim
        or not alias_prim.IsValid()
    ):
        return
    canonical_prim = _canonical_material_prim(composed_prim)
    canonical_path = str(canonical_prim.GetPath())
    members = bindings.setdefault(
        canonical_path,
        _MaterialBindingMembers(material_prim=canonical_prim),
    )
    members.material_alias_paths.add(str(alias_prim.GetPath()))


def _material_library_owner_path(material_path: str) -> str | None:
    """Return the prim path that owns a material library scope, if any."""
    parts = [part for part in material_path.split("/") if part]
    for index in range(len(parts) - 1, 0, -1):
        if parts[index].lower() in {"looks", "materials"}:
            return "/" + "/".join(parts[:index])
    return None


def _path_has_prefix(path: str, prefix: str) -> bool:
    """Return True when ``path`` equals or descends from ``prefix``."""
    return Sdf.Path(path).HasPrefix(Sdf.Path(prefix))


def _should_fold_subset_material_into_parent(
    *,
    subset_material: UsdShade.Material,
    parent_material: UsdShade.Material,
    subset_path: str,
    default_prim_path: str | None,
) -> bool:
    """Identify component-local subset material clones of an upstream material."""
    if not subset_material or not parent_material:
        return False
    subset_prim = subset_material.GetPrim()
    parent_prim = parent_material.GetPrim()
    if (
        not subset_prim
        or not subset_prim.IsValid()
        or not parent_prim
        or not parent_prim.IsValid()
    ):
        return False

    subset_material_path = str(subset_prim.GetPath())
    parent_material_path = str(parent_prim.GetPath())
    if subset_material_path == parent_material_path:
        return False

    subset_owner = _material_library_owner_path(subset_material_path)
    if subset_owner is None:
        return False
    owner_depth = len([part for part in subset_owner.split("/") if part])
    if owner_depth < 2:
        return False
    if default_prim_path and subset_owner == default_prim_path:
        return False
    parent_owner = _material_library_owner_path(parent_material_path)
    if parent_owner == subset_owner:
        return False

    subset_parent = str(Sdf.Path(subset_path).GetParentPath())
    return _path_has_prefix(subset_parent, subset_owner)


def _build_effective_material_binding_index(
    stage: Usd.Stage,
) -> tuple[
    dict[str, _MaterialBindingMembers],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, _MaterialCandidate],
]:
    """Build effective whole-prim and subset bindings in one scene traversal."""
    bindings: dict[str, _MaterialBindingMembers] = {}
    renderable_prim_paths: set[str] = set()
    renderable_subset_paths: set[str] = set()
    material_candidates: dict[str, _MaterialCandidate] = {}
    default_prim = stage.GetDefaultPrim()
    default_prim_path = (
        str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else None
    )

    for prim in _iter_renderable_prims(stage):
        if prim.IsA(UsdShade.Material):
            canonical_prim = _canonical_material_prim(prim)
            canonical_path = str(canonical_prim.GetPath())
            candidate = material_candidates.setdefault(
                canonical_path,
                _MaterialCandidate(material_prim=canonical_prim),
            )
            candidate.alias_paths.add(str(prim.GetPath()))

        if not prim.IsA(UsdGeom.Gprim):
            continue

        prim_path = str(prim.GetPath())
        renderable_prim_paths.add(prim_path)
        binding_api = UsdShade.MaterialBindingAPI(prim)

        material, _ = binding_api.ComputeBoundMaterial()
        _add_material_binding(bindings, material, prim_path=prim_path)

        for subset in binding_api.GetMaterialBindSubsets():
            subset_prim = subset.GetPrim()
            subset_path = str(subset_prim.GetPath())
            renderable_subset_paths.add(subset_path)
            subset_material, _ = UsdShade.MaterialBindingAPI(
                subset_prim
            ).ComputeBoundMaterial()
            if _should_fold_subset_material_into_parent(
                subset_material=subset_material,
                parent_material=material,
                subset_path=subset_path,
                default_prim_path=default_prim_path,
            ):
                _add_material_binding(bindings, material, subset_path=subset_path)
                _add_material_alias(bindings, material, subset_material)
                continue
            _add_material_binding(
                bindings,
                subset_material,
                subset_path=subset_path,
            )

    return (
        bindings,
        tuple(sorted(renderable_prim_paths)),
        tuple(sorted(renderable_subset_paths)),
        material_candidates,
    )


def _build_material_bound_prim_index(stage: Usd.Stage) -> dict[str, list[str]]:
    """Build a one-pass index of material path to bound geometry prim paths."""
    bindings, _, _, _ = _build_effective_material_binding_index(stage)
    return {
        min(members.material_alias_paths, default=material_path): sorted(
            members.prim_paths
        )
        for material_path, members in bindings.items()
        if members.prim_paths
    }


def _material_info_from_prim(
    prim: Usd.Prim,
    *,
    prim_path: str | None = None,
    bound_prim_paths: list[str] | None = None,
    bound_subset_paths: list[str] | None = None,
    material_alias_paths: list[str] | None = None,
) -> MaterialInfo:
    """Extract a MaterialInfo from one composed or prototype material prim."""
    base_color = _read_color3f(prim, "inputs:base_color")
    if base_color is None:
        base_color = _read_shader_color(prim)
    base_color_texture, has_existing_texture = _find_existing_texture_paths(prim)
    base_metalness = _read_float(prim, "inputs:base_metalness")
    if base_metalness is None:
        base_metalness = _read_shader_float(prim, _METALNESS_INPUTS)
    specular_roughness = _read_float(prim, "inputs:specular_roughness")
    if specular_roughness is None:
        specular_roughness = _read_shader_float(prim, _ROUGHNESS_INPUTS)

    return MaterialInfo(
        prim_path=prim_path or str(prim.GetPath()),
        name=prim.GetName(),
        bound_prim_paths=sorted(bound_prim_paths or []),
        bound_subset_paths=sorted(bound_subset_paths or []),
        base_color=base_color or (0.5, 0.5, 0.5),
        base_color_texture=base_color_texture,
        base_metalness=base_metalness,
        specular_roughness=specular_roughness,
        has_existing_texture=has_existing_texture,
        material_alias_paths=sorted(material_alias_paths or []),
    )


def discover_materials(
    stage: Usd.Stage,
    prim_paths: list[str] | None = None,
) -> list[MaterialInfo]:
    """Discover materials in a USD stage.

    Traverses the stage to find Material prims, extracts their constant
    OpenPBR properties plus common MaterialX/MDL shader-network metadata, and
    identifies which geometry prims use each material.

    Args:
        stage: An open USD stage.
        prim_paths: Optional list of material prim paths to restrict to.
            If None, all materials in the stage are discovered.

    Returns:
        List of MaterialInfo for each discovered material.
    """
    materials: list[MaterialInfo] = []
    bound_prims_by_material = _build_material_bound_prim_index(stage)

    # Collect material prims -- use TraverseAll to include 'over' prims
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdShade.Material):
            continue

        mat_path = str(prim.GetPath())

        # Apply filter if specified
        if prim_paths and mat_path not in prim_paths:
            continue

        info = _material_info_from_prim(
            prim,
            bound_prim_paths=bound_prims_by_material.get(mat_path, []),
        )
        materials.append(info)

        logger.info(
            "Discovered material: %s (base_color=%s, has_texture=%s, bound_prims=%d)",
            info.name,
            info.base_color,
            info.has_existing_texture,
            len(info.bound_prim_paths),
        )

    materials.sort(key=lambda material: material.prim_path)
    logger.info("Discovered %d materials total", len(materials))
    return materials


def _normalize_scope_paths(
    paths: list[str] | tuple[str, ...] | None,
    *,
    argument_name: str,
) -> tuple[str, ...]:
    """Validate, deduplicate, and sort absolute USD prim scope paths."""
    if not paths:
        return ()
    normalized: set[str] = set()
    for raw_path in paths:
        path = Sdf.Path(raw_path)
        if not path.IsAbsolutePath() or not path.IsPrimPath() or str(path) == "/":
            raise ValueError(
                f"{argument_name} must contain absolute USD prim paths: {raw_path!r}"
            )
        normalized.add(str(path))
    return tuple(sorted(normalized))


def _path_is_in_scope(path: str, scopes: tuple[str, ...]) -> bool:
    """Return whether a path equals or descends from one of the scopes."""
    if not scopes:
        return True
    sdf_path = Sdf.Path(path)
    return any(sdf_path.HasPrefix(Sdf.Path(scope)) for scope in scopes)


def _filter_member_paths(
    paths: set[str],
    *,
    prim_scope_paths: tuple[str, ...],
    upstream_assignment_paths: tuple[str, ...],
) -> list[str]:
    """Apply explicit and upstream prim scopes to binding member paths."""
    return sorted(
        path
        for path in paths
        if _path_is_in_scope(path, prim_scope_paths)
        and _path_is_in_scope(path, upstream_assignment_paths)
    )


def _copy_material_with_members(
    material: MaterialInfo,
    *,
    bound_prim_paths: list[str],
    bound_subset_paths: list[str],
) -> MaterialInfo:
    """Copy immutable discovery properties with scoped binding membership."""
    return MaterialInfo(
        prim_path=material.prim_path,
        name=material.name,
        bound_prim_paths=bound_prim_paths,
        bound_subset_paths=bound_subset_paths,
        base_color=material.base_color,
        base_color_texture=material.base_color_texture,
        base_metalness=material.base_metalness,
        specular_roughness=material.specular_roughness,
        has_existing_texture=material.has_existing_texture,
        material_alias_paths=list(material.material_alias_paths),
    )


def discover_effective_materials(
    stage: Usd.Stage,
    material_prim_paths: list[str] | tuple[str, ...] | None = None,
    prim_scope_paths: list[str] | tuple[str, ...] | None = None,
    upstream_assignment_paths: list[str] | tuple[str, ...] | None = None,
) -> EffectiveMaterialDiscovery:
    """Discover authored materials and their effective renderable bindings.

    Material scope paths select exact material prims. Prim and upstream
    assignment scope paths select the named prim and all descendants. When both
    prim scopes are supplied, a binding member must be in both scopes. This
    function performs no model/backend work and is safe to use during planning.
    """
    material_scopes = _normalize_scope_paths(
        material_prim_paths,
        argument_name="material_prim_paths",
    )
    prim_scopes = _normalize_scope_paths(
        prim_scope_paths,
        argument_name="prim_scope_paths",
    )
    assignment_scopes = _normalize_scope_paths(
        upstream_assignment_paths,
        argument_name="upstream_assignment_paths",
    )

    (
        binding_identities,
        all_prim_paths,
        all_subset_paths,
        material_candidates,
    ) = _build_effective_material_binding_index(stage)

    # TraverseAll preserves compatibility with authored typed-over materials;
    # the indexed candidates contribute materials visible only through instance
    # proxies, including unbound materials. Canonical identity makes the union
    # deterministic before a stable composed path is selected for each result.
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdShade.Material):
            continue
        canonical_prim = _canonical_material_prim(prim)
        canonical_path = str(canonical_prim.GetPath())
        candidate = material_candidates.setdefault(
            canonical_path,
            _MaterialCandidate(material_prim=canonical_prim),
        )
        candidate.alias_paths.add(str(prim.GetPath()))

    material_prims: dict[str, Usd.Prim] = {}
    material_aliases: dict[str, set[str]] = {}
    bindings: dict[str, _MaterialBindingMembers] = {}
    for identity, candidate in material_candidates.items():
        binding_members = binding_identities.get(identity)
        aliases = set(candidate.alias_paths)
        if binding_members is not None:
            aliases.update(binding_members.material_alias_paths)
        material_path = min(candidate.alias_paths, default=identity)
        material_prims[material_path] = candidate.material_prim
        material_aliases[material_path] = aliases | {material_path}
        if binding_members is not None:
            bindings[material_path] = binding_members

    authored_materials: list[MaterialInfo] = []
    for material_path in sorted(material_prims):
        scoped_members = bindings.get(material_path)
        authored_materials.append(
            _material_info_from_prim(
                material_prims[material_path],
                prim_path=material_path,
                bound_prim_paths=(
                    sorted(scoped_members.prim_paths)
                    if scoped_members is not None
                    else []
                ),
                bound_subset_paths=(
                    sorted(scoped_members.subset_paths)
                    if scoped_members is not None
                    else []
                ),
                material_alias_paths=sorted(material_aliases[material_path]),
            )
        )

    effective_materials: list[MaterialInfo] = []
    skipped_materials: list[MaterialDiscoverySkip] = []
    for material in authored_materials:
        scoped_members = bindings.get(material.prim_path)
        aliases = material_aliases.get(material.prim_path, {material.prim_path})
        if material_scopes and not aliases.intersection(material_scopes):
            skipped_materials.append(
                MaterialDiscoverySkip(
                    material_prim_path=material.prim_path,
                    material_name=material.name,
                    reason_code="outside_material_scope",
                    reason="Material is outside the explicit material scope.",
                )
            )
            continue

        if scoped_members is None or not (
            scoped_members.prim_paths or scoped_members.subset_paths
        ):
            skipped_materials.append(
                MaterialDiscoverySkip(
                    material_prim_path=material.prim_path,
                    material_name=material.name,
                    reason_code="not_effectively_bound",
                    reason="Material has no effective renderable binding.",
                )
            )
            continue

        explicit_prims = _filter_member_paths(
            scoped_members.prim_paths,
            prim_scope_paths=prim_scopes,
            upstream_assignment_paths=(),
        )
        explicit_subsets = _filter_member_paths(
            scoped_members.subset_paths,
            prim_scope_paths=prim_scopes,
            upstream_assignment_paths=(),
        )
        if prim_scopes and not (explicit_prims or explicit_subsets):
            skipped_materials.append(
                MaterialDiscoverySkip(
                    material_prim_path=material.prim_path,
                    material_name=material.name,
                    reason_code="outside_prim_scope",
                    reason="Material has no binding in the explicit prim scope.",
                )
            )
            continue

        scoped_prims = _filter_member_paths(
            scoped_members.prim_paths,
            prim_scope_paths=prim_scopes,
            upstream_assignment_paths=assignment_scopes,
        )
        scoped_subsets = _filter_member_paths(
            scoped_members.subset_paths,
            prim_scope_paths=prim_scopes,
            upstream_assignment_paths=assignment_scopes,
        )
        if assignment_scopes and not (scoped_prims or scoped_subsets):
            skipped_materials.append(
                MaterialDiscoverySkip(
                    material_prim_path=material.prim_path,
                    material_name=material.name,
                    reason_code="outside_upstream_assignment_scope",
                    reason=(
                        "Material has no binding in the upstream assignment scope."
                    ),
                )
            )
            continue

        effective_materials.append(
            _copy_material_with_members(
                material,
                bound_prim_paths=scoped_prims,
                bound_subset_paths=scoped_subsets,
            )
        )

    renderable_prim_paths = tuple(
        path
        for path in all_prim_paths
        if _path_is_in_scope(path, prim_scopes)
        and _path_is_in_scope(path, assignment_scopes)
    )
    renderable_subset_paths = tuple(
        path
        for path in all_subset_paths
        if _path_is_in_scope(path, prim_scopes)
        and _path_is_in_scope(path, assignment_scopes)
    )

    return EffectiveMaterialDiscovery(
        authored_materials=tuple(authored_materials),
        effective_materials=tuple(effective_materials),
        renderable_prim_paths=renderable_prim_paths,
        renderable_subset_paths=renderable_subset_paths,
        skipped_materials=tuple(skipped_materials),
    )


@dataclass
class PrimTextureUnit:
    """One texture-generation unit: a specific prim getting a specific texture.

    In per-material mode, prim_path is empty and key equals the material name.
    In per-prim mode, each bound prim gets its own unit with a unique key.
    """

    prim_path: str
    """Geometry prim path (e.g., '/World/Rail_L'). Empty in per-material mode."""

    material_info: MaterialInfo
    """The original shared material bound to this prim."""

    key: str
    """Unique key for dict lookups (e.g., 'Aluminum_Brushed__Rail_L')."""

    prompt: str
    """Text prompt for this unit's texture."""

    opacity: float
    """Blend opacity."""

    detail_policy: str = "default"
    """Texture detail policy for this unit."""

    seed: int | None = None
    """Seed for reproducibility. Different seeds per prim yield unique textures."""


def _stable_hash(s: str) -> int:
    """Deterministic hash stable across Python processes (unlike builtin hash)."""
    import hashlib

    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)


def _sanitize_prim_name(prim_path: str) -> str:
    """Extract a filesystem/USD-safe name from a prim path."""
    leaf = prim_path.rsplit("/", 1)[-1]
    return leaf.replace(" ", "_").replace("-", "_")


def resolve_material_texture_spec(
    material: MaterialInfo,
    material_textures: dict[str, dict],
) -> tuple[str, dict] | None:
    """Resolve a material spec by canonical path, aliases, then display name."""
    keys = [material.prim_path, *material.material_alias_paths, material.name]
    for key in dict.fromkeys(keys):
        spec = material_textures.get(key)
        if isinstance(spec, dict):
            return key, spec
    return None


def expand_to_prim_units(
    materials: list[MaterialInfo],
    material_textures: dict[str, dict],
    mode: str = "per_material",
    default_detail_policy: str = "default",
) -> list[PrimTextureUnit]:
    """Expand materials into texture generation units.

    Args:
        materials: Discovered materials with bound prim info.
        material_textures: Per-material texture specs from config.
        mode: "per_material" (one texture per material) or
              "per_prim" (unique texture per geometry prim).
        default_detail_policy: Global texture detail policy used when a
            material spec does not set one explicitly.

    Returns:
        List of PrimTextureUnit, one per generation job.
    """
    units: list[PrimTextureUnit] = []

    def _prompt_value(value: object, *, config_key: str) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        raise ValueError(f"{config_key} must be a string")

    def _unit_key_prefix(mat: MaterialInfo, spec_key: str) -> str:
        identity = spec_key if spec_key != mat.name else mat.name
        return identity.strip("/").replace("/", "_") or mat.name

    for mat in materials:
        spec_item = resolve_material_texture_spec(mat, material_textures)
        if spec_item is None:
            continue
        spec_key, spec = spec_item
        unit_key_prefix = _unit_key_prefix(mat, spec_key)

        base_prompt = _prompt_value(
            spec.get("prompt", ""),
            config_key=f"material_textures.{spec_key}.prompt",
        )
        base_opacity = spec.get("opacity", 0.85)
        base_detail_policy = normalize_detail_policy(
            spec.get("detail_policy"),
            config_key=f"material_textures.{spec_key}.detail_policy",
            default=default_detail_policy,
        )

        if mode == "per_prim" and mat.bound_prim_paths:
            # One unit per bound prim
            per_prim_overrides = spec.get("per_prim", {})

            # Detect leaf name collisions within this material
            leaf_names = [_sanitize_prim_name(p) for p in mat.bound_prim_paths]
            has_collision = len(leaf_names) != len(set(leaf_names))

            for prim_path in mat.bound_prim_paths:
                leaf = _sanitize_prim_name(prim_path)

                # Use full sanitized path if leaf names collide
                if has_collision:
                    safe_name = prim_path.strip("/").replace("/", "_")
                else:
                    safe_name = leaf

                key = f"{unit_key_prefix}__{safe_name}"

                # Check for per-prim overrides (by full path or leaf name)
                override = (
                    per_prim_overrides.get(prim_path)
                    or per_prim_overrides.get(leaf)
                    or {}
                )

                prompt = _prompt_value(
                    override.get("prompt", base_prompt),
                    config_key=(
                        f"material_textures.{spec_key}.per_prim.{prim_path}.prompt"
                    ),
                )
                opacity = override.get("opacity", base_opacity)
                detail_policy = normalize_detail_policy(
                    override.get("detail_policy"),
                    config_key=(
                        f"material_textures.{spec_key}.per_prim."
                        f"{prim_path}.detail_policy"
                    ),
                    default=base_detail_policy,
                )
                prompt = apply_detail_policy_to_prompt(prompt, detail_policy)
                seed = _stable_hash(prim_path)

                units.append(
                    PrimTextureUnit(
                        prim_path=prim_path,
                        material_info=mat,
                        key=key,
                        prompt=prompt,
                        opacity=opacity,
                        detail_policy=detail_policy,
                        seed=seed,
                    )
                )
        else:
            # Per-material mode: one unit per material
            prompt = apply_detail_policy_to_prompt(base_prompt, base_detail_policy)
            units.append(
                PrimTextureUnit(
                    prim_path="",
                    material_info=mat,
                    key=unit_key_prefix,
                    prompt=prompt,
                    opacity=base_opacity,
                    detail_policy=base_detail_policy,
                )
            )

    return units


def discover_materials_from_file(
    usd_path: str | Path,
    prim_paths: list[str] | None = None,
) -> list[MaterialInfo]:
    """Convenience wrapper that opens a USD file and discovers materials.

    Args:
        usd_path: Path to the USD file.
        prim_paths: Optional material prim path filter.

    Returns:
        List of MaterialInfo.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")
    return discover_materials(stage, prim_paths)


def discover_effective_materials_from_file(
    usd_path: str | Path,
    material_prim_paths: list[str] | tuple[str, ...] | None = None,
    prim_scope_paths: list[str] | tuple[str, ...] | None = None,
    upstream_assignment_paths: list[str] | tuple[str, ...] | None = None,
) -> EffectiveMaterialDiscovery:
    """Open a USD file and return deterministic effective material discovery."""
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")
    return discover_effective_materials(
        stage,
        material_prim_paths=material_prim_paths,
        prim_scope_paths=prim_scope_paths,
        upstream_assignment_paths=upstream_assignment_paths,
    )
