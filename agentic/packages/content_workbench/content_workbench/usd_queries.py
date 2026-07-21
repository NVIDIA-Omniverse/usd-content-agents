# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only USD query helpers for scene inspection sessions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import (
    DiagnosticRecord,
    MaterialBindingResponse,
    PropertiesResponse,
    TreeChild,
    TreeResponse,
)

_REMOTE_SCHEMES = frozenset({"http", "https", "omniverse", "s3"})


def is_remote_uri(value: str) -> bool:
    """Return True when a string looks like a remote asset URI."""
    parsed = urlparse(value)
    return parsed.scheme.lower() in _REMOTE_SCHEMES


def _usd_modules() -> tuple[Any, Any, Any, Any, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    return Gf, Sdf, Usd, UsdGeom, UsdShade


def _jsonable(value: Any, *, max_items: int = 32) -> Any:
    """Convert common USD values into JSON-safe values."""
    _gf, sdf, _usd, _usd_geom, _usd_shade = _usd_modules()

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, sdf.AssetPath):
        return {"path": value.path, "resolved_path": value.resolvedPath}
    if isinstance(value, sdf.Path):
        return str(value)
    if hasattr(value, "real") and hasattr(value, "imag"):
        return [value.real, value.imag]
    if hasattr(value, "__len__") and not isinstance(value, dict):
        try:
            size = len(value)
        except Exception:
            size = None
        if size is not None and size > max_items:
            return {"type": type(value).__name__, "size": size}
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, max_items=max_items)
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_jsonable(item, max_items=max_items) for item in value]
    if hasattr(value, "__iter__") and not isinstance(value, bytes):
        try:
            return [_jsonable(item, max_items=max_items) for item in list(value)]
        except Exception:
            pass
    return str(value)


def _asset_strings(value: Any) -> list[str]:
    """Extract string/asset values that can point at external assets."""
    _gf, sdf, _usd, _usd_geom, _usd_shade = _usd_modules()

    if value is None:
        return []
    if isinstance(value, sdf.AssetPath):
        return [item for item in [value.path, value.resolvedPath] if item]
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_asset_strings(item))
        return values
    if isinstance(value, tuple | list):
        values = []
        for item in value:
            values.extend(_asset_strings(item))
        return values
    if hasattr(value, "__iter__") and not isinstance(value, bytes):
        values = []
        try:
            for item in value:
                values.extend(_asset_strings(item))
        except Exception:
            return []
        return values
    return []


def _prim_children(prim: Any, usd: Any) -> list[Any]:
    """Return direct children, including children exposed through instances."""
    try:
        return list(prim.GetFilteredChildren(usd.TraverseInstanceProxies()))
    except Exception:
        return list(prim.GetChildren())


class UsdSceneQueries:
    """Read-only wrapper around a USD stage."""

    def __init__(self, scene_path: Path) -> None:
        _gf, _sdf, usd, _usd_geom, _usd_shade = _usd_modules()
        self.scene_path = scene_path
        self.stage = usd.Stage.Open(str(scene_path))
        if self.stage is None:
            raise ValueError(f"Failed to open USD stage: {scene_path}")

    def close(self) -> None:
        """Release the held USD stage reference."""
        self.stage = None

    def has_prim(self, prim_path: str) -> bool:
        """Return whether a prim exists on the stage."""
        return self.stage.GetPrimAtPath(prim_path).IsValid()

    def root_prim_path(self) -> str:
        """Resolve a useful hierarchy root without assuming /World."""
        world = self.stage.GetPrimAtPath("/World")
        if world.IsValid():
            return "/World"

        default_prim = self.stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            return str(default_prim.GetPath())

        for child in self.stage.GetPseudoRoot().GetChildren():
            return str(child.GetPath())

        return "/"

    def prim_count(self) -> int:
        """Return the number of prims in the stage."""
        return sum(1 for _prim in self.stage.TraverseAll())

    def get_children(self, prim_path: str | None = None) -> TreeResponse:
        """Return direct children for a prim."""
        _gf, _sdf, usd, _usd_geom, _usd_shade = _usd_modules()
        target_path = prim_path or self.root_prim_path()
        prim = self.stage.GetPrimAtPath(target_path)
        if not prim.IsValid():
            raise KeyError(f"Prim not found: {target_path}")

        children = [
            TreeChild(
                name=child.GetName(),
                path=str(child.GetPath()),
                type_name=child.GetTypeName() or "",
                active=child.IsActive(),
                loaded=child.IsLoaded(),
                children=bool(_prim_children(child, usd)),
            )
            for child in _prim_children(prim, usd)
        ]
        return TreeResponse(prim_path=target_path, children=children)

    def get_properties(
        self, prim_path: str, *, max_attributes: int = 100
    ) -> PropertiesResponse:
        """Return compact, JSON-safe properties for a prim."""
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"Prim not found: {prim_path}")

        attrs: dict[str, Any] = {}
        truncated = False
        for index, attr in enumerate(prim.GetAttributes()):
            if index >= max_attributes:
                truncated = True
                break
            name = attr.GetName()
            value = None
            if attr.HasValue():
                try:
                    value = attr.Get()
                except Exception as exc:
                    value = f"<unreadable: {exc}>"
            attrs[name] = {
                "type_name": str(attr.GetTypeName()) if attr.GetTypeName() else "",
                "value": _jsonable(value),
            }

        relationships = {
            rel.GetName(): [str(target) for target in rel.GetTargets()]
            for rel in prim.GetRelationships()
        }

        properties = {
            "path": str(prim.GetPath()),
            "name": prim.GetName(),
            "type_name": prim.GetTypeName() or "",
            "active": prim.IsActive(),
            "loaded": prim.IsLoaded(),
            "metadata": _jsonable(prim.GetAllMetadata()),
            "attributes": attrs,
            "relationships": relationships,
            "bounds": self.get_bounds(prim_path),
        }
        return PropertiesResponse(
            prim_path=prim_path,
            properties=properties,
            truncated=truncated,
        )

    def get_bounds(self, prim_path: str) -> dict[str, Any] | None:
        """Return world-aligned bounds for a prim when available."""
        _gf, _sdf, usd, usd_geom, _usd_shade = _usd_modules()
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"Prim not found: {prim_path}")

        try:
            cache = usd_geom.BBoxCache(
                usd.TimeCode.Default(),
                [usd_geom.Tokens.default_, usd_geom.Tokens.render],
                useExtentsHint=True,
            )
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if box.IsEmpty():
                return None
            min_v = box.GetMin()
            max_v = box.GetMax()
            values = [*min_v, *max_v]
            if not all(math.isfinite(float(value)) for value in values):
                return None
            return {
                "min": [float(value) for value in min_v],
                "max": [float(value) for value in max_v],
                "center": [float(value) for value in ((min_v + max_v) * 0.5)],
            }
        except Exception:
            return None

    def get_material_binding(self, prim_path: str) -> MaterialBindingResponse:
        """Return direct/inherited material binding information for a prim."""
        _gf, _sdf, _usd, _usd_geom, usd_shade = _usd_modules()
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"Prim not found: {prim_path}")

        binding_api = usd_shade.MaterialBindingAPI(prim)
        direct_rel = binding_api.GetDirectBindingRel()
        direct_targets = [str(target) for target in direct_rel.GetTargets()]

        bound_material_path = None
        relationship_path = None
        binding_source_path = None
        binding_type = "none"
        try:
            material, relationship = binding_api.ComputeBoundMaterial()
        except Exception:
            material = None
            relationship = None

        if material and material.GetPrim().IsValid():
            bound_material_path = str(material.GetPath())
            binding_type = "direct" if direct_targets else "inherited"
        if relationship and relationship.IsValid():
            relationship_path = str(relationship.GetPath())
            source_prim = relationship.GetPrim()
            if source_prim and source_prim.IsValid():
                binding_source_path = str(source_prim.GetPath())
                if binding_source_path == prim_path:
                    binding_type = "direct"
                elif bound_material_path:
                    binding_type = "inherited"

        return MaterialBindingResponse(
            prim_path=prim_path,
            binding_type=binding_type,
            bound_material_path=bound_material_path,
            binding_source_path=binding_source_path,
            relationship_path=relationship_path,
            direct_targets=direct_targets,
        )

    def expand_to_mesh_paths(self, prim_paths: list[str]) -> list[str]:
        """Expand prims or subtrees to concrete mesh prim paths."""
        _gf, _sdf, usd, usd_geom, _usd_shade = _usd_modules()
        mesh_paths: list[str] = []
        seen: set[str] = set()
        for prim_path in prim_paths:
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            for item in usd.PrimRange(prim, usd.TraverseInstanceProxies()):
                if item.IsA(usd_geom.Mesh):
                    path = str(item.GetPath())
                    if path not in seen:
                        mesh_paths.append(path)
                        seen.add(path)
        return mesh_paths

    def diagnostics(self) -> list[DiagnosticRecord]:
        """Return basic offline diagnostics for remote dependencies."""
        records: list[DiagnosticRecord] = []
        seen: set[tuple[str, str | None, str | None, str | None]] = set()

        for layer in self.stage.GetUsedLayers():
            identifier = layer.identifier
            if identifier and is_remote_uri(identifier):
                key = ("remote_layer", identifier, None, identifier)
                if key not in seen:
                    seen.add(key)
                    records.append(
                        DiagnosticRecord(
                            type="remote_layer",
                            source=identifier,
                            layer=identifier,
                        )
                    )

        for prim in self.stage.TraverseAll():
            prim_path = str(prim.GetPath())
            for attr in prim.GetAttributes():
                value = None
                try:
                    if attr.HasValue():
                        value = attr.Get()
                except Exception:
                    continue
                for source in _asset_strings(value):
                    if not is_remote_uri(source):
                        continue
                    key = ("remote_asset", source, prim_path, attr.GetName())
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        DiagnosticRecord(
                            type="remote_asset",
                            source=source,
                            prim_path=prim_path,
                            attribute=attr.GetName(),
                        )
                    )

        return records
