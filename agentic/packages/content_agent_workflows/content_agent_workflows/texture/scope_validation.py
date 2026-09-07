# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed USD checks for target-only Texture workflow edits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pxr import Ar, Sdf, Usd, UsdGeom, UsdShade
from pydantic import BaseModel, ConfigDict, Field, model_validator
from world_understanding.utils.usd.package import (
    USD_TEXTURE_EXTENSIONS,
    parse_package_member_asset_path,
)

from .models import TexturePlanDocument

TEXTURE_SCOPE_INVARIANT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-scope-invariants.v1"
] = "content-agent-workflows.texture-scope-invariants.v1"

_MAX_TEXTURE_RELOCATION_BYTES = 512 * 1024 * 1024
_MAX_TEXTURE_STAGE_READ_BYTES = 512 * 1024 * 1024
_TEXTURE_RELOCATION_READ_BYTES = 1024 * 1024


class TextureScopeViolation(BaseModel):
    """One deterministic target-scope or structural invariant violation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    prim_path: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class TextureScopeInvariantReport(BaseModel):
    """Comparison of a textured stage against its source stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.texture-scope-invariants.v1"] = (
        TEXTURE_SCOPE_INVARIANT_SCHEMA_VERSION
    )
    source_asset_path: str
    output_asset_path: str
    target_material_paths: tuple[str, ...]
    target_member_prim_paths: tuple[str, ...]
    passed: bool
    geometry_unchanged: bool
    non_target_materials_unchanged: bool
    bindings_unchanged: bool
    structure_unchanged_outside_target: bool
    violations: tuple[TextureScopeViolation, ...]

    @model_validator(mode="after")
    def _validate_passed(self) -> Self:
        expected = (
            self.geometry_unchanged
            and self.non_target_materials_unchanged
            and self.bindings_unchanged
            and self.structure_unchanged_outside_target
            and not self.violations
        )
        if self.passed is not expected:
            raise ValueError("passed must reflect all invariant checks")
        return self


def _unit_paths(
    plan: TexturePlanDocument,
    field_name: str,
) -> tuple[str, ...]:
    paths: list[str] = []
    for unit in plan.selected_units:
        raw_paths = getattr(unit, field_name, None)
        if raw_paths is None and unit.model_extra:
            raw_paths = unit.model_extra.get(field_name)
        if raw_paths is None:
            continue
        if not isinstance(raw_paths, list | tuple):
            raise ValueError(f"Texture plan {field_name} must be an array")
        paths.extend(str(path) for path in raw_paths)
    return tuple(dict.fromkeys(paths))


def _unit_value(unit: Any, field_name: str) -> Any:
    value = getattr(unit, field_name, None)
    if value is None and unit.model_extra:
        value = unit.model_extra.get(field_name)
    return value


def _unit_mode(unit: Any) -> str:
    return str(_unit_value(unit, "unit_mode") or "per_material")


@dataclass(frozen=True)
class _InternalInstanceSourceBinding:
    """One unique direct internal instance and its authorable source namespace."""

    owner_path: str
    source_root_path: str
    source_prim: Usd.Prim


def _instance_source_binding(
    stage: Usd.Stage,
    prim: Usd.Prim,
) -> _InternalInstanceSourceBinding | None:
    """Resolve one proxy through a unique direct root-layer internal reference."""

    cursor = prim
    while cursor.IsValid() and not cursor.IsInstance():
        cursor = cursor.GetParent()
    if not cursor.IsValid():
        return None
    prototype = cursor.GetPrototype()
    if not prototype:
        return None
    instance_paths = tuple(
        sorted(str(instance.GetPath()) for instance in prototype.GetInstances())
    )
    owner_path = str(cursor.GetPath())
    if instance_paths != (owner_path,):
        return None
    relative = prim.GetPath().MakeRelativePath(cursor.GetPath())
    root_layer = stage.GetRootLayer()
    query = Usd.PrimCompositionQuery.GetDirectReferences(cursor)
    matches: dict[
        tuple[str, str, str],
        _InternalInstanceSourceBinding,
    ] = {}
    for arc in query.GetCompositionArcs():
        node = arc.GetTargetNode()
        if node.layerStack.identifier.rootLayer != root_layer:
            continue
        source_root = stage.GetPrimAtPath(node.path)
        source_prim = stage.GetPrimAtPath(node.path.AppendPath(relative))
        if (
            not source_root.IsValid()
            or source_root.IsInstanceProxy()
            or not source_prim.IsValid()
            or source_prim.IsInstanceProxy()
        ):
            continue
        binding = _InternalInstanceSourceBinding(
            owner_path=owner_path,
            source_root_path=str(source_root.GetPath()),
            source_prim=source_prim,
        )
        matches[
            (
                binding.owner_path,
                binding.source_root_path,
                str(binding.source_prim.GetPath()),
            )
        ] = binding
    if len(matches) != 1:
        return None
    return next(iter(matches.values()))


def _instance_source_prim(stage: Usd.Stage, prim: Usd.Prim) -> Usd.Prim | None:
    """Resolve one unique internal instance proxy to its authorable source."""

    binding = _instance_source_binding(stage, prim)
    return binding.source_prim if binding is not None else None


def _authorable_instance_path(stage: Usd.Stage, raw_path: Any) -> str:
    """Return the source namespace path used for one exact proxy edit."""

    path = str(raw_path)
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.IsInstanceProxy():
        source = _instance_source_prim(stage, prim)
        if source is not None:
            return str(source.GetPath())
    return path


def _unit_clone_material_paths(
    unit: Any,
    stage: Usd.Stage,
) -> tuple[str, ...]:
    raw_paths = _unit_value(unit, "material_prim_paths")
    if not isinstance(raw_paths, list | tuple):
        return ()
    raw_aliases = _unit_value(unit, "material_alias_paths")
    aliases = raw_aliases if isinstance(raw_aliases, list | tuple) else ()
    for raw_path in dict.fromkeys((*raw_paths, *aliases)):
        candidate = stage.GetPrimAtPath(str(raw_path))
        if candidate.IsInstanceProxy():
            candidate = _instance_source_prim(stage, candidate) or Usd.Prim()
        if (
            candidate.IsValid()
            and not candidate.IsInstanceProxy()
            and not candidate.IsPrototype()
            and not candidate.IsInPrototype()
            and candidate.IsA(UsdShade.Material)
        ):
            clone_path = candidate.GetPath().GetParentPath().AppendChild(unit.unit_id)
            return (str(clone_path),)
    return tuple(
        str(Sdf.Path(str(path)).GetParentPath().AppendChild(unit.unit_id))
        for path in raw_paths
    )


def _mutable_output_material_paths(
    plan: TexturePlanDocument,
    source_stage: Usd.Stage,
    output_stage: Usd.Stage,
) -> tuple[str, ...]:
    paths: list[str] = []
    for unit in plan.selected_units:
        raw_paths = _unit_value(unit, "material_prim_paths")
        if not isinstance(raw_paths, list | tuple):
            continue
        source_clone_paths = _unit_clone_material_paths(unit, source_stage)
        source_collisions = [
            path
            for path in source_clone_paths
            if source_stage.GetPrimAtPath(path).IsValid()
        ]
        if source_collisions:
            raise ValueError(
                "Texture clone material paths already exist in the source: "
                + ", ".join(source_collisions)
            )
        output_clone_paths = _unit_clone_material_paths(unit, output_stage)
        output_has_clone = any(
            output_stage.GetPrimAtPath(path).IsValid() for path in output_clone_paths
        )
        if _unit_mode(unit) == "per_material" and not output_has_clone:
            paths.extend(str(path) for path in raw_paths)
            continue
        paths.extend(output_clone_paths)
    return tuple(dict.fromkeys(paths))


def _expected_member_rebindings(
    plan: TexturePlanDocument,
    output_stage: Usd.Stage,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for unit in plan.selected_units:
        clone_paths = _unit_clone_material_paths(unit, output_stage)
        if _unit_mode(unit) == "per_material" and not any(
            output_stage.GetPrimAtPath(path).IsValid() for path in clone_paths
        ):
            continue
        if not clone_paths:
            continue
        for field_name in ("member_prim_paths", "member_subset_paths"):
            raw_paths = _unit_value(unit, field_name)
            if isinstance(raw_paths, list | tuple):
                for path in raw_paths:
                    result[_authorable_instance_path(output_stage, path)] = tuple(
                        dict.fromkeys(clone_paths)
                    )
    return result


def _expected_rebinding_for_path(
    path: str,
    expected_rebindings: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    matching_roots = [
        root
        for root in expected_rebindings
        if path == root or path.startswith(f"{root}/")
    ]
    if not matching_roots:
        return ()
    root = max(matching_roots, key=len)
    return expected_rebindings[root]


def _effective_material_path(prim: Usd.Prim) -> str | None:
    material, _binding_relationship = UsdShade.MaterialBindingAPI(
        prim
    ).ComputeBoundMaterial()
    return str(material.GetPath()) if material else None


def _is_at_or_below(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _is_target_container_type_promotion(
    *,
    path: str,
    source_prim: Usd.Prim,
    output_prim: Usd.Prim,
    target_material_paths: tuple[str, ...],
) -> bool:
    """Allow the Texture Agent's typeless-to-Scope material container repair."""

    return (
        source_prim.GetTypeName() == ""
        and output_prim.GetTypeName() == "Scope"
        and any(
            target_path.startswith(f"{path}/") for target_path in target_material_paths
        )
    )


def _authored_metadata_state(
    usd_object: Any,
    *,
    ignored_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return composed metadata opinions without schema fallback metadata."""

    return {
        key: _canonical_metadata_value(value)
        for key, value in sorted(usd_object.GetAllAuthoredMetadata().items())
        if key not in ignored_keys
    }


def _canonical_metadata_value(value: Any) -> Any:
    """Convert USD metadata values to deterministic JSON-safe state."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonical_metadata_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_canonical_metadata_value(item) for item in value]
    return {
        "type": type(value).__name__,
        "value": str(value),
    }


def _has_authored_property(property_: Usd.Property) -> bool:
    """Distinguish authored property specs from built-in schema properties."""

    return bool(property_.GetPropertyStack())


class _TextureContentBudgetExceeded(RuntimeError):
    """Raised when unique texture bytes exceed one stage's validation budget."""


class _TextureContentCache:
    """Bound and deduplicate texture reads within one top-level computation."""

    def __init__(self) -> None:
        self._digests: dict[tuple[str, str], str | None] = {}
        self._bytes_read_by_stage: dict[str, int] = {}
        self._companion_indexes: dict[
            str,
            dict[tuple[str, str], tuple[Sdf.AssetPath, ...]],
        ] = {}

    def _remaining_bytes(self, stage_key: str) -> int:
        return _MAX_TEXTURE_STAGE_READ_BYTES - self._bytes_read_by_stage.get(
            stage_key,
            0,
        )

    def _record_read(self, stage_key: str, size: int) -> None:
        self._bytes_read_by_stage[stage_key] = (
            self._bytes_read_by_stage.get(stage_key, 0) + size
        )

    def local_sha256(self, path: Path, *, stage_key: str) -> str | None:
        identity = ("local", str(path))
        if identity in self._digests:
            return self._digests[identity]
        result = _read_local_texture_sha256(
            path,
            max_bytes=min(
                _MAX_TEXTURE_RELOCATION_BYTES,
                self._remaining_bytes(stage_key),
            ),
        )
        if result is None:
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            if size > self._remaining_bytes(stage_key):
                raise _TextureContentBudgetExceeded(
                    f"Texture content read budget exceeded for stage {stage_key}"
                )
            self._digests[identity] = None
            return None
        digest, size = result
        self._record_read(stage_key, size)
        self._digests[identity] = digest
        return digest

    def package_sha256(
        self,
        resolved_path: str,
        *,
        stage_key: str,
    ) -> str | None:
        identity = ("package", resolved_path)
        if identity in self._digests:
            return self._digests[identity]
        result = _read_ar_asset_sha256(
            resolved_path,
            max_bytes=min(
                _MAX_TEXTURE_RELOCATION_BYTES,
                self._remaining_bytes(stage_key),
            ),
        )
        if result is None:
            asset_size = _ar_asset_size(resolved_path)
            if asset_size is not None and asset_size > self._remaining_bytes(stage_key):
                raise _TextureContentBudgetExceeded(
                    f"Texture content read budget exceeded for stage {stage_key}"
                )
            self._digests[identity] = None
            return None
        digest, size = result
        self._record_read(stage_key, size)
        self._digests[identity] = digest
        return digest

    def companion_values(
        self,
        attribute: Usd.Attribute,
        raw: str,
    ) -> tuple[Sdf.AssetPath, ...]:
        stage = attribute.GetPrim().GetStage()
        stage_key = _stage_cache_key(stage)
        index = self._companion_indexes.get(stage_key)
        if index is None:
            mutable_index: dict[tuple[str, str], dict[str, Sdf.AssetPath]] = {}
            for prim in stage.Traverse():
                material_path = _enclosing_material_path(prim)
                if material_path is None:
                    continue
                for candidate in prim.GetAttributes():
                    value = candidate.Get(Usd.TimeCode.Default())
                    if not isinstance(value, Sdf.AssetPath) or not value.path:
                        continue
                    key = (material_path, Path(value.path).name)
                    mutable_index.setdefault(key, {}).setdefault(
                        value.resolvedPath,
                        value,
                    )
            index = {
                key: tuple(values.values()) for key, values in mutable_index.items()
            }
            self._companion_indexes[stage_key] = index
        material_path = _enclosing_material_path(attribute.GetPrim())
        if material_path is None:
            return ()
        return index.get((material_path, Path(raw).name), ())


def _stage_cache_key(stage: Usd.Stage) -> str:
    return str(stage.GetRootLayer().identifier)


def _enclosing_material_path(prim: Usd.Prim) -> str | None:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if current.IsA(UsdShade.Material):
            return str(current.GetPath())
        current = current.GetParent()
    return None


def _canonical_attribute_value(
    value: Any,
    *,
    content_cache: _TextureContentCache,
    stage_key: str,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> Any:
    """Keep authored Asset identity stable while binding it to texture content."""

    if isinstance(value, Sdf.Path):
        return (
            path_canonicalizer(value) if path_canonicalizer is not None else str(value)
        )
    if isinstance(value, Sdf.AssetPath):
        state = {"asset_path": value.path}
        content_sha256 = _asset_path_content_sha256(
            value,
            content_cache=content_cache,
            stage_key=stage_key,
        )
        if content_sha256 is not None:
            state["content_sha256"] = content_sha256
        return state
    if isinstance(value, Sdf.AssetPathArray):
        return [
            _canonical_attribute_value(
                item,
                content_cache=content_cache,
                stage_key=stage_key,
                path_canonicalizer=path_canonicalizer,
            )
            for item in value
        ]
    return repr(value)


def _relocation_normalized_texture_asset_state(value: Any) -> Any:
    """Remove only the package-local location from content-bound textures."""

    if isinstance(value, dict):
        if set(value) == {"asset_path", "content_sha256"}:
            asset_path = value["asset_path"]
            content_sha256 = value["content_sha256"]
            if isinstance(asset_path, str) and isinstance(content_sha256, str):
                return {
                    "asset_name": Path(asset_path).name,
                    "content_sha256": content_sha256,
                }
        return {
            key: _relocation_normalized_texture_asset_state(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relocation_normalized_texture_asset_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_relocation_normalized_texture_asset_state(item) for item in value)
    return value


def _attribute_state(
    attribute: Usd.Attribute,
    *,
    content_cache: _TextureContentCache,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> dict[str, Any]:
    time_samples = attribute.GetTimeSamples()
    property_stack = attribute.GetPropertyStack(Usd.TimeCode.Default())
    has_authored_default = any(spec.HasDefaultValue() for spec in property_stack)
    stage_key = _stage_cache_key(attribute.GetPrim().GetStage())
    values: list[tuple[str, Any]] = []
    if has_authored_default:
        values.append(
            (
                "default",
                _canonical_attribute_value(
                    attribute.Get(Usd.TimeCode.Default()),
                    content_cache=content_cache,
                    stage_key=stage_key,
                    path_canonicalizer=path_canonicalizer,
                ),
            )
        )
    values.extend(
        (
            str(time),
            _canonical_attribute_value(
                attribute.Get(time),
                content_cache=content_cache,
                stage_key=stage_key,
                path_canonicalizer=path_canonicalizer,
            ),
        )
        for time in time_samples
    )
    return {
        "type": str(attribute.GetTypeName()),
        "values": values,
        "connections": tuple(
            (path_canonicalizer(path) if path_canonicalizer is not None else str(path))
            for path in attribute.GetConnections()
        ),
        "metadata": _authored_metadata_state(
            attribute,
            ignored_keys=("typeName",),
        ),
    }


def _relationship_state(
    relationship: Usd.Relationship,
    *,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> dict[str, Any]:
    return {
        "targets": tuple(
            (path_canonicalizer(path) if path_canonicalizer is not None else str(path))
            for path in relationship.GetTargets()
        ),
        "metadata": _authored_metadata_state(relationship),
    }


def _property_state(
    prim: Usd.Prim,
    *,
    content_cache: _TextureContentCache,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in prim.GetAttributes():
        if _has_authored_property(attribute):
            result[attribute.GetName()] = {
                "kind": "attribute",
                **_attribute_state(
                    attribute,
                    content_cache=content_cache,
                    path_canonicalizer=path_canonicalizer,
                ),
            }
    for relationship in prim.GetRelationships():
        if _has_authored_property(relationship):
            result[relationship.GetName()] = {
                "kind": "relationship",
                **_relationship_state(
                    relationship,
                    path_canonicalizer=path_canonicalizer,
                ),
            }
    return result


def _resolved_local_texture_path(
    attribute: Usd.Attribute,
    value: Any,
    *,
    allowed_extensions: frozenset[str],
) -> tuple[str, Path] | None:
    """Resolve one local texture using its strongest authoring layer."""

    raw = value.path if isinstance(value, Sdf.AssetPath) else value
    if not isinstance(raw, str) or not raw or "://" in raw:
        return None
    if Path(raw).suffix.lower() not in allowed_extensions:
        return None

    resolved = (
        value.resolvedPath
        if isinstance(value, Sdf.AssetPath) and value.resolvedPath
        else ""
    )
    if resolved:
        candidate = Path(resolved)
    else:
        property_stack = attribute.GetPropertyStack(Usd.TimeCode.Default())
        if not property_stack:
            return None
        layer_path = property_stack[0].layer.realPath
        if not layer_path:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(layer_path).parent / candidate

    try:
        candidate = candidate.resolve(strict=True)
    except (OSError, ValueError):
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in allowed_extensions:
        return None
    return raw, candidate


def _is_safe_loose_texture_path(
    texture_path: Path,
    *,
    layer_asset_path: Path,
    allow_sibling_textures: bool,
) -> bool:
    """Require a texture below the layer directory or its sibling texture tree."""

    try:
        layer_parent = layer_asset_path.parent.resolve(strict=True)
    except (OSError, ValueError):
        return False
    allowed_roots = [layer_parent]
    if allow_sibling_textures:
        try:
            allowed_roots.append(
                (layer_parent.parent / "textures").resolve(strict=True)
            )
        except (OSError, ValueError):
            pass
    for allowed_root in allowed_roots:
        try:
            texture_path.relative_to(allowed_root)
        except ValueError:
            continue
        return True
    return False


def _read_local_texture_sha256(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[str, int] | None:
    """Read and hash one local texture within a caller-provided byte limit."""

    digest = hashlib.sha256()
    try:
        size = path.stat().st_size
        if size < 0 or size > max_bytes:
            return None
        with path.open("rb") as stream:
            bytes_read = 0
            for chunk in iter(
                lambda: stream.read(_TEXTURE_RELOCATION_READ_BYTES),
                b"",
            ):
                bytes_read += len(chunk)
                if bytes_read > size or bytes_read > max_bytes:
                    return None
                digest.update(chunk)
    except OSError:
        return None
    return (digest.hexdigest(), size) if bytes_read == size else None


def _ar_asset_size(resolved_path: str) -> int | None:
    try:
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(resolved_path))
        if asset is None:
            return None
        size = asset.GetSize()
    except Exception:
        return None
    return size if isinstance(size, int) and size >= 0 else None


def _read_ar_asset_sha256(
    resolved_path: str,
    *,
    max_bytes: int,
) -> tuple[str, int] | None:
    """Read and hash one resolver asset within a caller-provided byte limit."""

    try:
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(resolved_path))
        if asset is None:
            return None
        size = asset.GetSize()
        if not isinstance(size, int) or size < 0 or size > max_bytes:
            return None
        digest = hashlib.sha256()
        offset = 0
        while offset < size:
            read_size = min(_TEXTURE_RELOCATION_READ_BYTES, size - offset)
            chunk = asset.Read(read_size, offset)
            if not isinstance(chunk, bytes) or len(chunk) != read_size:
                return None
            digest.update(chunk)
            offset += read_size
    except Exception:
        return None
    return digest.hexdigest(), size


def _package_asset_texture_sha256(
    value: Sdf.AssetPath,
    *,
    expected_package_path: Path | None,
    content_cache: _TextureContentCache,
    stage_key: str,
) -> str | None:
    """Hash one safe texture member resolved from the expected USDZ."""

    raw = value.path
    resolved = value.resolvedPath
    if (
        not raw
        or not resolved
        or Path(raw).suffix.lower() not in USD_TEXTURE_EXTENSIONS
        or Path(raw).is_absolute()
    ):
        return None
    package_member = parse_package_member_asset_path(resolved)
    if package_member is None:
        return None
    package_path, member_name = package_member
    if expected_package_path is not None:
        try:
            if package_path.resolve(strict=True) != expected_package_path.resolve(
                strict=True
            ):
                return None
        except (OSError, ValueError):
            return None
    if (
        Path(member_name).suffix.lower() not in USD_TEXTURE_EXTENSIONS
        or Path(member_name).name != Path(raw).name
    ):
        return None
    return content_cache.package_sha256(
        resolved,
        stage_key=stage_key,
    )


def _asset_path_content_sha256(
    value: Sdf.AssetPath,
    *,
    content_cache: _TextureContentCache,
    stage_key: str,
) -> str | None:
    """Hash a resolved local or packaged texture without resolver location."""

    packaged_digest = _package_asset_texture_sha256(
        value,
        expected_package_path=None,
        content_cache=content_cache,
        stage_key=stage_key,
    )
    if packaged_digest is not None:
        return packaged_digest
    if (
        not value.resolvedPath
        or Path(value.path).suffix.lower() not in USD_TEXTURE_EXTENSIONS
    ):
        return None
    try:
        resolved_path = Path(value.resolvedPath).resolve(strict=True)
    except (OSError, ValueError):
        return None
    if (
        not resolved_path.is_file()
        or resolved_path.suffix.lower() not in USD_TEXTURE_EXTENSIONS
    ):
        return None
    return content_cache.local_sha256(
        resolved_path,
        stage_key=stage_key,
    )


def _companion_package_png_sha256(
    attribute: Usd.Attribute,
    raw: str,
    *,
    expected_package_path: Path,
    expected_content_sha256: str | None,
    content_cache: _TextureContentCache,
) -> str | None:
    """Resolve a string texture through exact-package Asset dependencies."""

    if Path(raw).is_absolute() or Path(raw).suffix.lower() != ".png":
        return None
    matches: set[str] = set()
    stage_key = _stage_cache_key(attribute.GetPrim().GetStage())
    for value in content_cache.companion_values(attribute, raw):
        digest = _package_asset_texture_sha256(
            value,
            expected_package_path=expected_package_path,
            content_cache=content_cache,
            stage_key=stage_key,
        )
        if digest is None:
            continue
        if expected_content_sha256 is not None:
            if digest == expected_content_sha256:
                return digest
            continue
        matches.add(digest)
        if len(matches) > 1:
            return None
    return next(iter(matches)) if len(matches) == 1 else None


def _package_texture_sha256(
    attribute: Usd.Attribute,
    value: Any,
    *,
    expected_package_path: Path,
    content_cache: _TextureContentCache,
    expected_content_sha256: str | None = None,
) -> str | None:
    if isinstance(value, Sdf.AssetPath):
        return _package_asset_texture_sha256(
            value,
            expected_package_path=expected_package_path,
            content_cache=content_cache,
            stage_key=_stage_cache_key(attribute.GetPrim().GetStage()),
        )
    if isinstance(value, str):
        return _companion_package_png_sha256(
            attribute,
            value,
            expected_package_path=expected_package_path,
            expected_content_sha256=expected_content_sha256,
            content_cache=content_cache,
        )
    return None


def _is_content_equivalent_texture_relocation(
    *,
    source_attribute: Usd.Attribute,
    output_attribute: Usd.Attribute,
    source_asset_path: Path,
    output_asset_path: Path,
    content_cache: _TextureContentCache,
) -> bool:
    """Allow only content-equivalent texture relocation within the output bundle."""

    if source_attribute.GetTypeName() != output_attribute.GetTypeName():
        return False
    if source_attribute.GetConnections() != output_attribute.GetConnections():
        return False
    if _authored_metadata_state(
        source_attribute,
        ignored_keys=("typeName",),
    ) != _authored_metadata_state(
        output_attribute,
        ignored_keys=("typeName",),
    ):
        return False
    if source_attribute.GetTimeSamples() or output_attribute.GetTimeSamples():
        return False

    source_value = source_attribute.Get(Usd.TimeCode.Default())
    output_value = output_attribute.Get(Usd.TimeCode.Default())
    is_asset_reference = isinstance(source_value, Sdf.AssetPath) and isinstance(
        output_value,
        Sdf.AssetPath,
    )
    is_shader_string_reference = (
        isinstance(source_value, str)
        and isinstance(output_value, str)
        and source_attribute.GetPrim().IsA(UsdShade.Shader)
        and output_attribute.GetPrim().IsA(UsdShade.Shader)
        and source_attribute.GetName().startswith("inputs:")
        and source_attribute.GetName().endswith("_texture")
    )
    if not (is_asset_reference or is_shader_string_reference):
        return False
    allowed_extensions = (
        frozenset({".png"}) if is_shader_string_reference else USD_TEXTURE_EXTENSIONS
    )

    source_raw = (
        source_value.path if isinstance(source_value, Sdf.AssetPath) else source_value
    )
    output_raw = (
        output_value.path if isinstance(output_value, Sdf.AssetPath) else output_value
    )
    involves_package = (
        source_asset_path.suffix.lower() == ".usdz"
        or output_asset_path.suffix.lower() == ".usdz"
    )
    if (
        source_raw == output_raw
        and not involves_package
        and not is_shader_string_reference
    ) or Path(output_raw).is_absolute():
        return False
    if (
        is_shader_string_reference
        and output_asset_path.suffix.lower() == ".usdz"
        and source_raw != output_raw
        and output_raw != f"../textures/{Path(output_raw).name}"
    ):
        return False

    if source_asset_path.suffix.lower() == ".usdz":
        source_digest = _package_texture_sha256(
            source_attribute,
            source_value,
            expected_package_path=source_asset_path,
            content_cache=content_cache,
        )
    else:
        source_resolved = _resolved_local_texture_path(
            source_attribute,
            source_value,
            allowed_extensions=allowed_extensions,
        )
        if source_resolved is None:
            return False
        _, source_path = source_resolved
        if not _is_safe_loose_texture_path(
            source_path,
            layer_asset_path=source_asset_path,
            allow_sibling_textures=True,
        ):
            return False
        source_digest = content_cache.local_sha256(
            source_path,
            stage_key=_stage_cache_key(source_attribute.GetPrim().GetStage()),
        )

    if output_asset_path.suffix.lower() == ".usdz":
        output_digest = _package_texture_sha256(
            output_attribute,
            output_value,
            expected_package_path=output_asset_path,
            content_cache=content_cache,
            expected_content_sha256=source_digest,
        )
    else:
        output_resolved = _resolved_local_texture_path(
            output_attribute,
            output_value,
            allowed_extensions=allowed_extensions,
        )
        if output_resolved is None:
            return False
        _, output_path = output_resolved
        if source_raw == output_raw:
            if not _is_safe_loose_texture_path(
                output_path,
                layer_asset_path=output_asset_path,
                allow_sibling_textures=True,
            ):
                return False
        else:
            try:
                bundle_relative_path = output_path.relative_to(
                    output_asset_path.parent.parent.resolve(strict=True)
                )
            except (OSError, ValueError):
                return False
            if (
                len(bundle_relative_path.parts) < 2
                or bundle_relative_path.parts[0] != "textures"
            ):
                return False
        output_digest = content_cache.local_sha256(
            output_path,
            stage_key=_stage_cache_key(output_attribute.GetPrim().GetStage()),
        )

    return source_digest is not None and source_digest == output_digest


def _material_properties_unchanged(
    *,
    source_prim: Usd.Prim,
    output_prim: Usd.Prim,
    source_asset_path: Path,
    output_asset_path: Path,
    content_cache: _TextureContentCache,
    source_path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
    output_path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> bool:
    """Compare material properties with a narrow portable-texture exception."""

    source_state = _property_state(
        source_prim,
        content_cache=content_cache,
        path_canonicalizer=source_path_canonicalizer,
    )
    output_state = _property_state(
        output_prim,
        content_cache=content_cache,
        path_canonicalizer=output_path_canonicalizer,
    )
    if source_state.keys() != output_state.keys():
        return False

    for property_name in source_state:
        if source_state[property_name] == output_state[property_name]:
            if source_state[property_name].get("kind") != "attribute":
                continue
            source_attribute = source_prim.GetAttribute(property_name)
            output_attribute = output_prim.GetAttribute(property_name)
            source_value = source_attribute.Get(Usd.TimeCode.Default())
            output_value = output_attribute.Get(Usd.TimeCode.Default())
            requires_texture_content_check = (
                source_attribute.IsValid()
                and output_attribute.IsValid()
                and isinstance(source_value, str)
                and isinstance(output_value, str)
                and source_attribute.GetPrim().IsA(UsdShade.Shader)
                and output_attribute.GetPrim().IsA(UsdShade.Shader)
                and source_attribute.GetName().startswith("inputs:")
                and source_attribute.GetName().endswith("_texture")
                and bool(source_value)
                and bool(output_value)
                and "://" not in source_value
                and "://" not in output_value
                and Path(source_value).suffix.lower() == ".png"
                and Path(output_value).suffix.lower() == ".png"
            )
            if requires_texture_content_check and not (
                _is_content_equivalent_texture_relocation(
                    source_attribute=source_attribute,
                    output_attribute=output_attribute,
                    source_asset_path=source_asset_path,
                    output_asset_path=output_asset_path,
                    content_cache=content_cache,
                )
            ):
                return False
            continue
        if not (
            source_state[property_name].get("kind") == "attribute"
            and output_state[property_name].get("kind") == "attribute"
        ):
            return False
        source_attribute = source_prim.GetAttribute(property_name)
        output_attribute = output_prim.GetAttribute(property_name)
        if not source_attribute.IsValid() or not output_attribute.IsValid():
            return False
        if not _is_content_equivalent_texture_relocation(
            source_attribute=source_attribute,
            output_attribute=output_attribute,
            source_asset_path=source_asset_path,
            output_asset_path=output_asset_path,
            content_cache=content_cache,
        ):
            return False
    return True


def _geometry_state(
    prim: Usd.Prim,
    *,
    allow_texture_coordinates: bool,
    content_cache: _TextureContentCache,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if allow_texture_coordinates and (
            name == "primvars:st" or name.startswith("primvars:st:")
        ):
            continue
        if _has_authored_property(attribute):
            result[name] = _attribute_state(
                attribute,
                content_cache=content_cache,
                path_canonicalizer=path_canonicalizer,
            )
    return result


def _texture_coordinate_state(
    prim: Usd.Prim,
    *,
    content_cache: _TextureContentCache,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> dict[str, Any]:
    """Capture authored ``primvars:st`` state that texture work may change."""

    return {
        attribute.GetName(): _attribute_state(
            attribute,
            content_cache=content_cache,
            path_canonicalizer=path_canonicalizer,
        )
        for attribute in prim.GetAttributes()
        if (
            attribute.GetName() == "primvars:st"
            or attribute.GetName().startswith("primvars:st:")
        )
        and _has_authored_property(attribute)
    }


def _binding_state(
    prim: Usd.Prim,
    *,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        relationship.GetName(): _relationship_state(
            relationship,
            path_canonicalizer=path_canonicalizer,
        )
        for relationship in prim.GetRelationships()
        if relationship.GetName().startswith("material:binding")
        and _has_authored_property(relationship)
    }


def _is_expected_default_rebinding(
    *,
    source_bindings: dict[str, dict[str, Any]],
    output_bindings: dict[str, dict[str, Any]],
    output_prim: Usd.Prim,
    expected_material_paths: tuple[str, ...],
) -> bool:
    """Allow only the planned all-purpose direct binding target change."""

    if not expected_material_paths:
        return False
    effective_material_path = _effective_material_path(output_prim)
    if effective_material_path not in expected_material_paths:
        return False

    default_binding_name = "material:binding"
    output_default = output_bindings.get(default_binding_name)
    if output_default is None or output_default["targets"] != (
        effective_material_path,
    ):
        return False

    source_default = source_bindings.get(default_binding_name)
    if source_default is not None:
        allowed_metadata = output_default["metadata"] == source_default["metadata"]
    else:
        # A member may inherit its source material from an ancestor. Authoring
        # the planned unit-specific binding then creates a relationship whose
        # only metadata is USD's standard relationship property metadata.
        # Continue to reject arbitrary metadata authored with that new binding.
        allowed_metadata = output_default["metadata"] == {
            "custom": False,
            "variability": _canonical_metadata_value(Sdf.VariabilityUniform),
        }
    if not allowed_metadata:
        return False

    source_other = {
        name: state
        for name, state in source_bindings.items()
        if name != default_binding_name
    }
    output_other = {
        name: state
        for name, state in output_bindings.items()
        if name != default_binding_name
    }
    return source_other == output_other


def _outside_target_prim_state(
    prim: Usd.Prim,
    *,
    attributes_checked_elsewhere: bool,
    material_properties_checked_elsewhere: bool,
    allow_texture_coordinates: bool,
    allow_material_binding_api: bool,
    content_cache: _TextureContentCache,
    path_canonicalizer: Callable[[Sdf.Path], str] | None = None,
) -> dict[str, Any]:
    """Capture authored state not covered by geometry/material/binding checks."""

    metadata = _authored_metadata_state(prim, ignored_keys=("typeName",))
    if allow_material_binding_api:
        metadata.pop("apiSchemas", None)
        metadata["appliedSchemasWithoutMaterialBindingAPI"] = tuple(
            schema
            for schema in prim.GetAppliedSchemas()
            if schema != "MaterialBindingAPI"
        )

    properties: dict[str, Any] = {}
    if not material_properties_checked_elsewhere:
        if not attributes_checked_elsewhere:
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if allow_texture_coordinates and (
                    name == "primvars:st" or name.startswith("primvars:st:")
                ):
                    continue
                if _has_authored_property(attribute):
                    properties[name] = {
                        "kind": "attribute",
                        **_attribute_state(
                            attribute,
                            content_cache=content_cache,
                            path_canonicalizer=path_canonicalizer,
                        ),
                    }
        for relationship in prim.GetRelationships():
            name = relationship.GetName()
            if name.startswith("material:binding") or not _has_authored_property(
                relationship
            ):
                continue
            properties[name] = {
                "kind": "relationship",
                **_relationship_state(
                    relationship,
                    path_canonicalizer=path_canonicalizer,
                ),
            }
    return {
        "metadata": metadata,
        "properties": properties,
    }


def _stage_prims(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    return {str(prim.GetPath()): prim for prim in stage.Traverse()}


@dataclass(frozen=True)
class _InternalInstanceSourceNamespace:
    """Stable ownership for one unique root-layer internal instance source."""

    owner_path: str
    source_root_path: str
    normalize_generated_path: bool


@dataclass(frozen=True)
class _ScopedPrimEntry:
    """One prim keyed independently from USD's synthetic prototype root name."""

    actual_path: str
    display_path: str
    prim: Usd.Prim


_ScopedPrimKey = tuple[str, str, str]


def _is_generated_flattened_prototype_root(
    stage: Usd.Stage,
    source_root_path: str,
) -> bool:
    """Recognize only USD-flatten-generated synthetic prototype namespaces."""

    path = Sdf.Path(source_root_path)
    name = path.name
    suffix = name.removeprefix("Flattened_Prototype_")
    if (
        path.GetParentPath() != Sdf.Path.absoluteRootPath
        or not suffix.isdigit()
        or suffix.startswith("0")
    ):
        return False
    root_layer = stage.GetRootLayer()
    if not root_layer.documentation.startswith(
        "Generated from Composed Stage of root layer "
    ):
        return False
    root_spec = root_layer.GetPrimAtPath(path)
    return bool(
        root_spec is not None
        and root_spec.specifier == Sdf.SpecifierOver
        and set(root_spec.ListInfoKeys()) == {"specifier"}
    )


def _internal_instance_sources_for_scope(
    stage: Usd.Stage,
    plan: TexturePlanDocument,
) -> tuple[_InternalInstanceSourceNamespace, ...]:
    """Return only unambiguous owner-to-source namespaces selected by the plan."""

    raw_paths = tuple(
        dict.fromkeys(
            path
            for field_name in (
                "material_prim_paths",
                "member_prim_paths",
                "member_subset_paths",
            )
            for path in _unit_paths(plan, field_name)
        )
    )
    candidates: dict[
        tuple[str, str],
        _InternalInstanceSourceNamespace,
    ] = {}
    roots_by_owner: dict[str, set[str]] = {}
    owners_by_root: dict[str, set[str]] = {}
    for raw_path in raw_paths:
        prim = stage.GetPrimAtPath(raw_path)
        if not prim.IsValid() or not prim.IsInstanceProxy():
            continue
        binding = _instance_source_binding(stage, prim)
        if binding is None:
            continue
        namespace = _InternalInstanceSourceNamespace(
            owner_path=binding.owner_path,
            source_root_path=binding.source_root_path,
            normalize_generated_path=_is_generated_flattened_prototype_root(
                stage,
                binding.source_root_path,
            ),
        )
        candidates[(namespace.owner_path, namespace.source_root_path)] = namespace
        roots_by_owner.setdefault(namespace.owner_path, set()).add(
            namespace.source_root_path
        )
        owners_by_root.setdefault(namespace.source_root_path, set()).add(
            namespace.owner_path
        )
    return tuple(
        sorted(
            (
                namespace
                for namespace in candidates.values()
                if len(roots_by_owner[namespace.owner_path]) == 1
                and len(owners_by_root[namespace.source_root_path]) == 1
            ),
            key=lambda item: (item.owner_path, item.source_root_path),
        )
    )


def _scoped_path_canonicalizer(
    stage: Usd.Stage,
    plan: TexturePlanDocument,
) -> Callable[[Sdf.Path], str]:
    """Key synthetic source paths by owner without conflating public paths."""

    namespaces = tuple(
        namespace
        for namespace in _internal_instance_sources_for_scope(stage, plan)
        if namespace.normalize_generated_path
    )

    def canonicalize(path: Sdf.Path) -> str:
        if not path.IsAbsolutePath():
            return str(path)
        matches = tuple(
            namespace
            for namespace in namespaces
            if path.HasPrefix(Sdf.Path(namespace.source_root_path))
        )
        if not matches:
            return str(path)
        if len(matches) != 1:
            raise ValueError(
                "Texture instance-source path belongs to multiple owner namespaces: "
                f"{path}"
            )
        namespace = matches[0]
        relative = path.MakeRelativePath(Sdf.Path(namespace.source_root_path))
        return json.dumps(
            {
                "namespace": "instance-source",
                "owner_path": namespace.owner_path,
                "relative_path": str(relative),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    return canonicalize


def _stage_prims_for_scope(
    stage: Usd.Stage,
    plan: TexturePlanDocument,
) -> dict[str, Usd.Prim]:
    """Include complete unique-instance source trees omitted by Traverse()."""

    prims = _stage_prims(stage)
    for namespace in _internal_instance_sources_for_scope(stage, plan):
        source_root = stage.GetPrimAtPath(namespace.source_root_path)
        if not source_root.IsValid():
            continue
        cursor = source_root
        while cursor.IsValid() and not cursor.IsPseudoRoot():
            prims[str(cursor.GetPath())] = cursor
            cursor = cursor.GetParent()
        for descendant in Usd.PrimRange.AllPrims(source_root):
            if descendant.IsValid():
                prims[str(descendant.GetPath())] = descendant
    return prims


def _canonical_scoped_prims(
    stage: Usd.Stage,
    plan: TexturePlanDocument,
) -> dict[_ScopedPrimKey, _ScopedPrimEntry]:
    """Key unique internal source trees by owner and owner-relative path."""

    namespaces = tuple(
        namespace
        for namespace in _internal_instance_sources_for_scope(stage, plan)
        if namespace.normalize_generated_path
    )
    entries: dict[_ScopedPrimKey, _ScopedPrimEntry] = {}
    for actual_path, prim in _stage_prims_for_scope(stage, plan).items():
        matching_namespaces = tuple(
            namespace
            for namespace in namespaces
            if _is_at_or_below(actual_path, (namespace.source_root_path,))
        )
        if len(matching_namespaces) == 1:
            namespace = matching_namespaces[0]
            if actual_path == namespace.source_root_path:
                relative_path = ""
                display_path = namespace.owner_path
            else:
                relative = Sdf.Path(actual_path).MakeRelativePath(
                    Sdf.Path(namespace.source_root_path)
                )
                relative_path = str(relative)
                display_path = str(Sdf.Path(namespace.owner_path).AppendPath(relative))
            key: _ScopedPrimKey = (
                "instance-source",
                namespace.owner_path,
                relative_path,
            )
        else:
            key = ("path", actual_path, "")
            display_path = actual_path
        existing = entries.get(key)
        if existing is not None and existing.actual_path != actual_path:
            raise ValueError(
                "Texture instance-source canonical key collision at "
                f"{display_path}: {existing.actual_path}, {actual_path}"
            )
        entries[key] = _ScopedPrimEntry(
            actual_path=actual_path,
            display_path=display_path,
            prim=prim,
        )
    return entries


def _material_tree_keys(
    entries: dict[_ScopedPrimKey, _ScopedPrimEntry],
) -> set[_ScopedPrimKey]:
    material_roots = tuple(
        entry.actual_path
        for entry in entries.values()
        if entry.prim.IsA(UsdShade.Material)
    )
    return {
        key
        for key, entry in entries.items()
        if _is_at_or_below(entry.actual_path, material_roots)
    }


def _material_state(
    *,
    stage_prims: dict[str, Usd.Prim],
    material_paths: tuple[str, ...],
    content_cache: _TextureContentCache,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for material_path in material_paths:
        matching_paths = sorted(
            path
            for path in stage_prims
            if path == material_path or path.startswith(f"{material_path}/")
        )
        if not matching_paths:
            state[material_path] = {"present": False}
            continue
        state[material_path] = {
            "present": True,
            "prims": {
                path: {
                    "type": stage_prims[path].GetTypeName(),
                    "metadata": _authored_metadata_state(
                        stage_prims[path],
                        ignored_keys=("typeName",),
                    ),
                    "properties": _property_state(
                        stage_prims[path],
                        content_cache=content_cache,
                    ),
                }
                for path in matching_paths
            },
        }
    return state


def _member_binding_state(
    *,
    stage: Usd.Stage,
    stage_prims: dict[str, Usd.Prim],
    member_paths: tuple[str, ...],
    excluded_descendant_paths: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    state: dict[str, Any] = {}
    effective_material_paths: list[str] = []
    for member_root in member_paths:
        matching_paths = sorted(
            path
            for path in stage_prims
            if path == member_root or path.startswith(f"{member_root}/")
            if not _is_at_or_below(path, excluded_descendant_paths)
        )
        if not matching_paths:
            state[member_root] = {"present": False}
            continue
        for path in matching_paths:
            prim = stage_prims[path]
            effective_path = _effective_material_path(prim)
            if effective_path:
                effective_material_paths.append(effective_path)
            state[path] = {
                "present": True,
                "direct": _binding_state(prim),
                "effective_material_path": effective_path,
            }
    return state, tuple(dict.fromkeys(effective_material_paths))


def _member_texture_coordinate_state(
    *,
    stage_prims: dict[str, Usd.Prim],
    member_prim_paths: tuple[str, ...],
    member_subset_paths: tuple[str, ...],
    excluded_descendant_paths: tuple[str, ...],
    content_cache: _TextureContentCache,
) -> dict[str, Any]:
    """Capture only UV state owned by one selected unit.

    Prim selections own their descendants. Subset selections own the subset
    subtree plus the exact parent prim where USD texture coordinates are
    conventionally authored.
    """

    state: dict[str, Any] = {}
    recursive_roots = tuple(dict.fromkeys((*member_prim_paths, *member_subset_paths)))
    for member_root in recursive_roots:
        matching_paths = sorted(
            path
            for path in stage_prims
            if path == member_root or path.startswith(f"{member_root}/")
            if not _is_at_or_below(path, excluded_descendant_paths)
        )
        if not matching_paths:
            state[member_root] = {"present": False}
            continue
        for path in matching_paths:
            state[path] = {
                "present": True,
                "primvars_st": _texture_coordinate_state(
                    stage_prims[path],
                    content_cache=content_cache,
                ),
            }

    for parent_path in dict.fromkeys(
        str(Sdf.Path(path).GetParentPath()) for path in member_subset_paths
    ):
        parent_prim = stage_prims.get(parent_path)
        state[parent_path] = (
            {
                "present": True,
                "primvars_st": _texture_coordinate_state(
                    parent_prim,
                    content_cache=content_cache,
                ),
            }
            if parent_prim is not None
            else {"present": False}
        )
    return state


def texture_unit_material_state_digests(
    *,
    output_asset_path: str | Path,
    plan: TexturePlanDocument,
    unit_ids: tuple[str, ...],
    normalize_texture_asset_relocations: bool = False,
) -> dict[str, str]:
    """Hash each selected unit's composed material tree in one output stage.

    When relocation normalization is requested, a resolvable texture asset is
    represented by its basename and content digest. This preserves all other
    authored state while allowing content-equivalent package-local path rewrites.
    """

    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Texture material-state unit IDs must be unique")
    plan_units = {unit.unit_id: unit for unit in plan.selected_units}
    unknown_ids = sorted(set(unit_ids) - set(plan_units))
    if unknown_ids:
        raise ValueError(
            "Texture material-state unit IDs are outside the plan: "
            + ", ".join(unknown_ids)
        )

    output_path = Path(output_asset_path).expanduser().resolve()
    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Could not open textured USD stage: {output_path}")
    content_cache = _TextureContentCache()
    stage_prims = _stage_prims(stage)
    member_prim_paths_by_unit: dict[str, tuple[str, ...]] = {}
    member_subset_paths_by_unit: dict[str, tuple[str, ...]] = {}
    member_paths_by_unit: dict[str, tuple[str, ...]] = {}
    for planned_unit_id, planned_unit in plan_units.items():
        raw_member_prim_paths = _unit_value(planned_unit, "member_prim_paths")
        member_prim_paths = (
            tuple(dict.fromkeys(str(path) for path in raw_member_prim_paths))
            if isinstance(raw_member_prim_paths, list | tuple)
            else ()
        )
        raw_member_subset_paths = _unit_value(planned_unit, "member_subset_paths")
        member_subset_paths = (
            tuple(dict.fromkeys(str(path) for path in raw_member_subset_paths))
            if isinstance(raw_member_subset_paths, list | tuple)
            else ()
        )
        member_prim_paths_by_unit[planned_unit_id] = member_prim_paths
        member_subset_paths_by_unit[planned_unit_id] = member_subset_paths
        member_paths_by_unit[planned_unit_id] = tuple(
            dict.fromkeys((*member_prim_paths, *member_subset_paths))
        )

    digests: dict[str, str] = {}
    for unit_id in unit_ids:
        unit = plan_units[unit_id]
        raw_paths = _unit_value(unit, "material_prim_paths")
        if not isinstance(raw_paths, list | tuple) or not raw_paths:
            raise RuntimeError(
                f"Texture plan unit {unit_id} has no material_prim_paths"
            )
        material_paths = tuple(dict.fromkeys(str(path) for path in raw_paths))
        member_paths = member_paths_by_unit[unit_id]
        excluded_descendant_paths = tuple(
            dict.fromkeys(
                path
                for other_unit_id, other_member_paths in member_paths_by_unit.items()
                if other_unit_id != unit_id
                for path in other_member_paths
                if path not in member_paths
                and any(path.startswith(f"{root}/") for root in member_paths)
            )
        )
        binding_state, effective_material_paths = _member_binding_state(
            stage=stage,
            stage_prims=stage_prims,
            member_paths=member_paths,
            excluded_descendant_paths=excluded_descendant_paths,
        )
        state = {
            "materials": _material_state(
                stage_prims=stage_prims,
                material_paths=tuple(
                    dict.fromkeys((*material_paths, *effective_material_paths))
                ),
                content_cache=content_cache,
            ),
            "member_bindings": binding_state,
            "member_texture_coordinates": _member_texture_coordinate_state(
                stage_prims=stage_prims,
                member_prim_paths=member_prim_paths_by_unit[unit_id],
                member_subset_paths=member_subset_paths_by_unit[unit_id],
                excluded_descendant_paths=excluded_descendant_paths,
                content_cache=content_cache,
            ),
        }
        if normalize_texture_asset_relocations:
            state = _relocation_normalized_texture_asset_state(state)
        canonical = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digests[unit_id] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digests


def _validate_texture_scope_invariants(
    *,
    source_asset_path: str | Path,
    output_asset_path: str | Path,
    plan: TexturePlanDocument,
    normalized_source_asset_path: str | Path | None = None,
) -> TextureScopeInvariantReport:
    """Verify geometry, bindings, and non-target materials stayed unchanged."""

    source_path = Path(source_asset_path).expanduser().resolve()
    source_stage_path = (
        Path(normalized_source_asset_path).expanduser().resolve()
        if normalized_source_asset_path is not None
        else source_path
    )
    output_path = Path(output_asset_path).expanduser().resolve()
    source_stage = Usd.Stage.Open(str(source_stage_path))
    output_stage = Usd.Stage.Open(str(output_path))
    if source_stage is None:
        raise RuntimeError(f"Could not open source USD stage: {source_path}")
    if output_stage is None:
        raise RuntimeError(f"Could not open textured USD stage: {output_path}")
    content_cache = _TextureContentCache()

    target_material_paths = _unit_paths(plan, "material_prim_paths")
    mutable_output_material_paths = _mutable_output_material_paths(
        plan,
        source_stage,
        output_stage,
    )
    target_member_prim_paths = _unit_paths(plan, "member_prim_paths")
    target_member_subset_paths = _unit_paths(plan, "member_subset_paths")
    target_member_paths = tuple(
        dict.fromkeys((*target_member_prim_paths, *target_member_subset_paths))
    )
    target_subset_parent_paths = tuple(
        dict.fromkeys(
            str(Sdf.Path(path).GetParentPath()) for path in target_member_subset_paths
        )
    )
    target_authorable_member_paths = tuple(
        dict.fromkeys(
            _authorable_instance_path(output_stage, path)
            for path in target_member_paths
        )
    )
    target_authorable_subset_parent_paths = tuple(
        dict.fromkeys(
            _authorable_instance_path(output_stage, path)
            for path in target_subset_parent_paths
        )
    )
    expected_rebindings = _expected_member_rebindings(plan, output_stage)
    target_authorable_paths = tuple(
        dict.fromkeys((*mutable_output_material_paths, *expected_rebindings))
    )
    if not target_material_paths:
        raise RuntimeError(
            "Texture plan does not expose material_prim_paths for scope validation"
        )

    source_prims = _canonical_scoped_prims(source_stage, plan)
    output_prims = _canonical_scoped_prims(output_stage, plan)
    source_path_canonicalizer = _scoped_path_canonicalizer(source_stage, plan)
    output_path_canonicalizer = _scoped_path_canonicalizer(output_stage, plan)
    violations: list[TextureScopeViolation] = []
    structure_unchanged = True
    geometry_unchanged = True
    materials_unchanged = True
    bindings_unchanged = True

    if _authored_metadata_state(
        source_stage.GetPseudoRoot()
    ) != _authored_metadata_state(output_stage.GetPseudoRoot()):
        structure_unchanged = False
        violations.append(
            TextureScopeViolation(
                code="stage.metadata_changed",
                prim_path="/",
                summary="Authored stage metadata changed.",
            )
        )

    outside_source = {
        key: entry
        for key, entry in source_prims.items()
        if not _is_at_or_below(entry.actual_path, mutable_output_material_paths)
    }
    outside_output = {
        key: entry
        for key, entry in output_prims.items()
        if not _is_at_or_below(entry.actual_path, mutable_output_material_paths)
    }
    for key in sorted(set(outside_source) ^ set(outside_output)):
        structure_unchanged = False
        changed_entry = outside_source.get(key) or outside_output.get(key)
        if changed_entry is not None and (
            changed_entry.prim.IsA(UsdGeom.Gprim)
            or changed_entry.prim.IsA(UsdGeom.Xformable)
        ):
            geometry_unchanged = False
        violations.append(
            TextureScopeViolation(
                code="structure.changed_outside_target",
                prim_path=(
                    changed_entry.display_path if changed_entry is not None else "/"
                ),
                summary="Prim presence changed outside the selected material scope.",
            )
        )
    for key in sorted(set(outside_source) & set(outside_output)):
        source_entry = outside_source[key]
        output_entry = outside_output[key]
        if source_entry.prim.GetTypeName() != output_entry.prim.GetTypeName() and not (
            _is_target_container_type_promotion(
                path=output_entry.actual_path,
                source_prim=source_entry.prim,
                output_prim=output_entry.prim,
                target_material_paths=mutable_output_material_paths,
            )
        ):
            structure_unchanged = False
            if (
                source_entry.prim.IsA(UsdGeom.Gprim)
                or source_entry.prim.IsA(UsdGeom.Xformable)
                or output_entry.prim.IsA(UsdGeom.Gprim)
                or output_entry.prim.IsA(UsdGeom.Xformable)
            ):
                geometry_unchanged = False
            violations.append(
                TextureScopeViolation(
                    code="structure.type_changed_outside_target",
                    prim_path=output_entry.display_path,
                    summary="Prim type changed outside the selected material scope.",
                )
            )

    source_material_keys = _material_tree_keys(source_prims)
    output_material_keys = _material_tree_keys(output_prims)
    non_target_material_keys = {
        key
        for key in source_material_keys | output_material_keys
        if not (
            key in output_prims
            and _is_at_or_below(
                output_prims[key].actual_path,
                mutable_output_material_paths,
            )
        )
    }

    common_paths = sorted(set(source_prims) & set(output_prims))
    binding_violation_paths: set[str] = set()
    for key in common_paths:
        source_entry = source_prims[key]
        output_entry = output_prims[key]
        source_prim = source_entry.prim
        output_prim = output_entry.prim
        output_actual_path = output_entry.actual_path
        is_geometry_or_transform = source_prim.IsA(UsdGeom.Gprim) or source_prim.IsA(
            UsdGeom.Xformable
        )
        allow_texture_coordinates = (
            _is_at_or_below(
                output_actual_path,
                target_authorable_member_paths,
            )
            or output_actual_path in target_authorable_subset_parent_paths
        )
        if is_geometry_or_transform:
            source_geometry = _geometry_state(
                source_prim,
                allow_texture_coordinates=allow_texture_coordinates,
                content_cache=content_cache,
                path_canonicalizer=source_path_canonicalizer,
            )
            output_geometry = _geometry_state(
                output_prim,
                allow_texture_coordinates=allow_texture_coordinates,
                content_cache=content_cache,
                path_canonicalizer=output_path_canonicalizer,
            )
            if source_geometry != output_geometry:
                geometry_unchanged = False
                violations.append(
                    TextureScopeViolation(
                        code="geometry.changed",
                        prim_path=output_entry.display_path,
                        summary=(
                            "Geometry or transform properties changed outside the "
                            "allowed target texture-coordinate exception."
                        ),
                    )
                )
        source_bindings = _binding_state(
            source_prim,
            path_canonicalizer=source_path_canonicalizer,
        )
        output_bindings = _binding_state(
            output_prim,
            path_canonicalizer=output_path_canonicalizer,
        )
        expected_material_paths = _expected_rebinding_for_path(
            output_actual_path,
            expected_rebindings,
        )
        allowed_rebinding = _is_expected_default_rebinding(
            source_bindings=_binding_state(source_prim),
            output_bindings=_binding_state(output_prim),
            output_prim=output_prim,
            expected_material_paths=expected_material_paths,
        )
        if source_bindings != output_bindings:
            if not allowed_rebinding:
                bindings_unchanged = False
                binding_violation_paths.add(output_actual_path)
                violations.append(
                    TextureScopeViolation(
                        code="material_binding.changed",
                        prim_path=output_entry.display_path,
                        summary="Material binding targets changed.",
                    )
                )

        if not _is_at_or_below(output_actual_path, mutable_output_material_paths):
            material_properties_checked_elsewhere = key in non_target_material_keys
            source_outside_state = _outside_target_prim_state(
                source_prim,
                attributes_checked_elsewhere=is_geometry_or_transform,
                material_properties_checked_elsewhere=(
                    material_properties_checked_elsewhere
                ),
                allow_texture_coordinates=allow_texture_coordinates,
                allow_material_binding_api=allowed_rebinding,
                content_cache=content_cache,
                path_canonicalizer=source_path_canonicalizer,
            )
            output_outside_state = _outside_target_prim_state(
                output_prim,
                attributes_checked_elsewhere=is_geometry_or_transform,
                material_properties_checked_elsewhere=(
                    material_properties_checked_elsewhere
                ),
                allow_texture_coordinates=allow_texture_coordinates,
                allow_material_binding_api=allowed_rebinding,
                content_cache=content_cache,
                path_canonicalizer=output_path_canonicalizer,
            )
            target_container_specifier_promotion = (
                source_prim.GetSpecifier() == Sdf.SpecifierOver
                and output_prim.GetSpecifier() == Sdf.SpecifierDef
                and any(
                    target_path.startswith(f"{output_actual_path}/")
                    for target_path in target_authorable_paths
                )
            )
            if target_container_specifier_promotion:
                source_outside_state["metadata"].pop("specifier", None)
                output_outside_state["metadata"].pop("specifier", None)
            if source_outside_state != output_outside_state:
                structure_unchanged = False
                violations.append(
                    TextureScopeViolation(
                        code="scope.authored_state_changed_outside_target",
                        prim_path=output_entry.display_path,
                        summary=(
                            "Authored properties or metadata changed outside the "
                            "selected material scope."
                        ),
                    )
                )

    for member_path, expected_material_paths in expected_rebindings.items():
        # Flattened internal instance sources may sit below an ``over`` prim.
        # Stage.Traverse() intentionally omits that subtree even though the
        # exact source prim remains valid and is the writable binding target.
        output_prim = output_stage.GetPrimAtPath(member_path)
        if (
            not output_prim.IsValid()
            or _effective_material_path(output_prim) not in expected_material_paths
        ) and member_path not in binding_violation_paths:
            bindings_unchanged = False
            violations.append(
                TextureScopeViolation(
                    code="material_binding.target_unit_mismatch",
                    prim_path=member_path,
                    summary=(
                        "Selected member is not bound to its unit-specific "
                        "output material."
                    ),
                )
            )

    for key in sorted(non_target_material_keys):
        source_entry = source_prims.get(key)
        output_entry = output_prims.get(key)
        if (
            source_entry is None
            or output_entry is None
            or not _material_properties_unchanged(
                source_prim=source_entry.prim,
                output_prim=output_entry.prim,
                source_asset_path=source_path,
                output_asset_path=output_path,
                content_cache=content_cache,
                source_path_canonicalizer=source_path_canonicalizer,
                output_path_canonicalizer=output_path_canonicalizer,
            )
        ):
            materials_unchanged = False
            violations.append(
                TextureScopeViolation(
                    code="material.non_target_changed",
                    prim_path=(
                        output_entry.display_path
                        if output_entry is not None
                        else (
                            source_entry.display_path
                            if source_entry is not None
                            else "/"
                        )
                    ),
                    summary="A non-target material property or shader changed.",
                )
            )

    return TextureScopeInvariantReport(
        source_asset_path=str(source_path),
        output_asset_path=str(output_path),
        target_material_paths=target_material_paths,
        target_member_prim_paths=target_member_paths,
        passed=not violations,
        geometry_unchanged=geometry_unchanged,
        non_target_materials_unchanged=materials_unchanged,
        bindings_unchanged=bindings_unchanged,
        structure_unchanged_outside_target=structure_unchanged,
        violations=tuple(violations),
    )


def validate_texture_scope_invariants(
    *,
    source_asset_path: str | Path,
    output_asset_path: str | Path,
    plan: TexturePlanDocument,
    normalized_source_asset_path: str | Path | None = None,
) -> TextureScopeInvariantReport:
    """Verify Texture scope while reporting aggregate read-budget exhaustion."""

    try:
        return _validate_texture_scope_invariants(
            source_asset_path=source_asset_path,
            output_asset_path=output_asset_path,
            plan=plan,
            normalized_source_asset_path=normalized_source_asset_path,
        )
    except _TextureContentBudgetExceeded as exc:
        source_path = Path(source_asset_path).expanduser().resolve()
        output_path = Path(output_asset_path).expanduser().resolve()
        target_material_paths = _unit_paths(plan, "material_prim_paths")
        target_member_prim_paths = _unit_paths(plan, "member_prim_paths")
        target_member_subset_paths = _unit_paths(plan, "member_subset_paths")
        return TextureScopeInvariantReport(
            source_asset_path=str(source_path),
            output_asset_path=str(output_path),
            target_material_paths=target_material_paths,
            target_member_prim_paths=tuple(
                dict.fromkeys((*target_member_prim_paths, *target_member_subset_paths))
            ),
            passed=False,
            geometry_unchanged=False,
            non_target_materials_unchanged=False,
            bindings_unchanged=True,
            structure_unchanged_outside_target=False,
            violations=(
                TextureScopeViolation(
                    code="texture.content_read_budget_exceeded",
                    prim_path="/",
                    summary=str(exc),
                ),
            ),
        )


__all__ = [
    "TEXTURE_SCOPE_INVARIANT_SCHEMA_VERSION",
    "TextureScopeInvariantReport",
    "TextureScopeViolation",
    "texture_unit_material_state_digests",
    "validate_texture_scope_invariants",
]
