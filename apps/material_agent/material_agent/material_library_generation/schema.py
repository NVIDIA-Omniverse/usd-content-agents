# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schemas for generated material library packages."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_LIBRARY_ROOT = "/World/Looks"


_SLUG_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_]+")
_SLUG_UNDERSCORES = re.compile(r"_+")


def make_material_id(value: str) -> str:
    """Return a stable, lowercase identifier for material artifacts."""
    slug = value.strip().lower().replace("-", "_").replace(" ", "_")
    slug = _SLUG_INVALID_CHARS.sub("_", slug)
    slug = _SLUG_UNDERSCORES.sub("_", slug).strip("_")
    if not slug:
        raise ValueError("material id cannot be empty")
    return slug


def make_usd_identifier(value: str) -> str:
    """Return a USD-safe identifier for a material prim name."""
    identifier = value.strip().replace("-", "_").replace(" ", "_")
    identifier = _SLUG_INVALID_CHARS.sub("_", identifier)
    identifier = _SLUG_UNDERSCORES.sub("_", identifier).strip("_")
    if not identifier:
        raise ValueError("USD identifier cannot be empty")
    if identifier[0].isdigit():
        identifier = f"Material_{identifier}"
    return identifier


@dataclass(frozen=True)
class PBRHints:
    """Scalar PBR profile values that constrain or synthesize backend channels."""

    metallic: float = 0.0
    roughness: float = 0.5
    opacity: float = 1.0
    transmission: float = 0.0
    ior: float = 1.5
    thin_walled: bool = False

    def validate(self) -> None:
        for name, value in (
            ("metallic", self.metallic),
            ("roughness", self.roughness),
            ("opacity", self.opacity),
            ("transmission", self.transmission),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.ior <= 0.0:
            raise ValueError(f"ior must be positive, got {self.ior}")

    def to_dict(self) -> dict[str, float | bool]:
        data: dict[str, float | bool] = {
            "metallic": float(self.metallic),
            "roughness": float(self.roughness),
        }
        if self.opacity != 1.0:
            data["opacity"] = float(self.opacity)
        if self.transmission != 0.0:
            data["transmission"] = float(self.transmission)
        if self.ior != 1.5:
            data["ior"] = float(self.ior)
        if self.thin_walled:
            data["thin_walled"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PBRHints:
        """Build PBR hints from YAML-compatible data."""
        if not data:
            return cls()
        return cls(
            metallic=float(data.get("metallic", 0.0)),
            roughness=float(data.get("roughness", 0.5)),
            opacity=float(data.get("opacity", 1.0)),
            transmission=float(data.get("transmission", 0.0)),
            ior=float(data.get("ior", 1.5)),
            thin_walled=bool(data.get("thin_walled", False)),
        )


@dataclass(frozen=True)
class IntendedPart:
    """Semantic part hint recorded by the material planner."""

    semantic_label: str
    evidence: str = ""
    prim_path_hints: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.semantic_label.strip():
            raise ValueError("intended part semantic_label cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"semantic_label": self.semantic_label}
        if self.evidence:
            data["evidence"] = self.evidence
        if self.prim_path_hints:
            data["prim_path_hints"] = list(self.prim_path_hints)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntendedPart:
        """Build an intended-part hint from YAML-compatible data."""
        if not isinstance(data, dict):
            raise TypeError("intended_parts entries must be dictionaries")
        prim_path_hints = data.get("prim_path_hints", ())
        if isinstance(prim_path_hints, str):
            prim_path_hints = (prim_path_hints,)
        return cls(
            semantic_label=str(data.get("semantic_label", "")),
            evidence=str(data.get("evidence", "")),
            prim_path_hints=tuple(str(path) for path in prim_path_hints),
        )


@dataclass(frozen=True)
class MaterialRecipe:
    """Recipe for generating one material in an asset-specific library."""

    name: str
    description: str
    appearance_prompt: str
    id: str | None = None
    color: str | None = None
    material: str | None = None
    finish: str | None = None
    base_color_hint: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pbr_hints: PBRHints = field(default_factory=PBRHints)
    reference_image_uris: tuple[str, ...] = ()
    intended_parts: tuple[IntendedPart, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        source = self.id if self.id is not None else self.name
        object.__setattr__(self, "id", make_material_id(source))

    @property
    def material_id(self) -> str:
        return self.id or make_material_id(self.name)

    @property
    def usd_name(self) -> str:
        return make_usd_identifier(self.name)

    @property
    def binding(self) -> str:
        return f"{DEFAULT_LIBRARY_ROOT}/{self.usd_name}"

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("material recipe name cannot be empty")
        if not self.description.strip():
            raise ValueError(f"material recipe {self.name!r} needs a description")
        if not self.appearance_prompt.strip():
            raise ValueError(
                f"material recipe {self.name!r} needs an appearance_prompt"
            )
        if len(self.base_color_hint) != 3:
            raise ValueError("base_color_hint must contain exactly three values")
        for channel in self.base_color_hint:
            if not 0.0 <= channel <= 1.0:
                raise ValueError(
                    f"base_color_hint values must be in [0, 1], got {channel}"
                )
        self.pbr_hints.validate()
        for part in self.intended_parts:
            part.validate()

    def to_dict(self, texture_paths: dict[str, str] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.material_id,
            "name": self.name,
            "description": self.description,
            "appearance_prompt": self.appearance_prompt,
            "base_color_hint": list(self.base_color_hint),
            "pbr_hints": self.pbr_hints.to_dict(),
            "priority": self.priority,
        }
        if self.color:
            data["color"] = self.color
        if self.material:
            data["material"] = self.material
        if self.finish:
            data["finish"] = self.finish
        if self.reference_image_uris:
            data["reference_image_uris"] = list(self.reference_image_uris)
        if self.intended_parts:
            data["intended_parts"] = [part.to_dict() for part in self.intended_parts]
        if texture_paths:
            data["generated_textures"] = texture_paths
        return data

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], base_dir: str | Path | None = None
    ) -> MaterialRecipe:
        """Build a material recipe from YAML-compatible data."""
        if not isinstance(data, dict):
            raise TypeError("material recipe entries must be dictionaries")

        base_color_hint = data.get("base_color_hint", (0.5, 0.5, 0.5))
        if len(base_color_hint) != 3:
            raise ValueError("base_color_hint must contain exactly three values")

        reference_image_uris = data.get("reference_image_uris", ())
        if isinstance(reference_image_uris, str):
            reference_image_uris = (reference_image_uris,)
        resolved_refs = tuple(
            _resolve_local_uri(str(uri), base_dir) for uri in reference_image_uris
        )

        intended_parts = tuple(
            IntendedPart.from_dict(part) for part in data.get("intended_parts", ())
        )

        return cls(
            id=data.get("id"),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            appearance_prompt=str(data.get("appearance_prompt", "")),
            color=data.get("color"),
            material=data.get("material"),
            finish=data.get("finish"),
            base_color_hint=tuple(float(value) for value in base_color_hint),
            pbr_hints=PBRHints.from_dict(data.get("pbr_hints")),
            reference_image_uris=resolved_refs,
            intended_parts=intended_parts,
            priority=int(data.get("priority", 0)),
        )


@dataclass(frozen=True)
class MaterialGenerationPlan:
    """VLM-authored plan for generating an asset-specific material library."""

    materials: tuple[MaterialRecipe, ...]
    asset: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def validate(self) -> None:
        if not self.materials:
            raise ValueError(
                "material generation plan must contain at least one recipe"
            )
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for recipe in self.materials:
            recipe.validate()
            material_id = recipe.material_id
            normalized_name = recipe.name.strip().lower()
            if material_id in seen_ids:
                raise ValueError(f"duplicate material id: {material_id}")
            if normalized_name in seen_names:
                raise ValueError(f"duplicate material name: {recipe.name}")
            seen_ids.add(material_id)
            seen_names.add(normalized_name)

    def to_dict(
        self,
        texture_paths_by_material_id: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        texture_paths_by_material_id = texture_paths_by_material_id or {}
        data: dict[str, Any] = {
            "version": self.version,
            "asset": self.asset,
            "materials": [
                recipe.to_dict(texture_paths_by_material_id.get(recipe.material_id))
                for recipe in self.materials
            ],
        }
        return data

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], base_dir: str | Path | None = None
    ) -> MaterialGenerationPlan:
        """Build a generation plan from YAML-compatible data."""
        if not isinstance(data, dict):
            raise TypeError("material generation plan must be a dictionary")

        materials_data = data.get("materials", ())
        if not isinstance(materials_data, list | tuple):
            raise TypeError("material generation plan 'materials' must be a list")

        plan = cls(
            version=int(data.get("version", 1)),
            asset=dict(data.get("asset") or {}),
            materials=tuple(
                MaterialRecipe.from_dict(recipe, base_dir=base_dir)
                for recipe in materials_data
            ),
        )
        plan.validate()
        return plan


def _resolve_local_uri(uri: str, base_dir: str | Path | None) -> str:
    """Resolve relative local reference-image paths next to the plan file."""
    uri = uri.strip()
    parsed = urlparse(uri)
    if not uri or parsed.scheme or not base_dir:
        return uri
    path = Path(uri)
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())


@dataclass(frozen=True)
class TextureMapSet:
    """Paths to generated/synthesized texture maps."""

    albedo: Path
    normal: Path
    orm: Path

    def as_relative_dict(self, base_dir: Path) -> dict[str, str]:
        return {
            "albedo": self.albedo.relative_to(base_dir).as_posix(),
            "normal": self.normal.relative_to(base_dir).as_posix(),
            "orm": self.orm.relative_to(base_dir).as_posix(),
        }


@dataclass(frozen=True)
class GeneratedMaterial:
    """Generated material artifact ready for USD and manifest authoring."""

    recipe: MaterialRecipe
    textures: TextureMapSet
    prototype_source: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.recipe.name

    @property
    def description(self) -> str:
        return self.recipe.description

    @property
    def binding(self) -> str:
        return self.recipe.binding

    @property
    def prototype_name(self) -> str | None:
        if not self.prototype_source:
            return None
        value = self.prototype_source.get("name")
        return str(value) if value else None


@dataclass(frozen=True)
class GeneratedMaterialLibrary:
    """Paths and manifest data for a generated material library package."""

    package_dir: Path
    material_library_path: Path
    materials_manifest_path: Path
    generation_plan_path: Path | None
    materials: tuple[GeneratedMaterial, ...]

    @property
    def materials_data(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for material in self.materials:
            entry = {
                "name": material.name,
                "description": material.description,
                "binding": material.binding,
            }
            if material.prototype_source:
                entry["prototype_source"] = dict(material.prototype_source)
            entries.append(entry)
        return {
            "library_path": str(self.material_library_path),
            "entries": entries,
        }
