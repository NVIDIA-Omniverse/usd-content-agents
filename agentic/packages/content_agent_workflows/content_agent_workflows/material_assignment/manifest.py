# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-owned material-library manifest resolution.

``materials.yaml`` describes workflow choices; it is intentionally resolved here,
outside either scene-operation backend.  Executors receive only a validated library
path, material prim path, material name, and target paths.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MAX_MATERIAL_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MATERIAL_MANIFEST_ENTRIES = 10_000
MAX_MATERIAL_MANIFEST_STRING_CHARS = 64 * 1024
MAX_MATERIAL_MANIFEST_TAGS_PER_ENTRY = 1024


@dataclass(frozen=True, slots=True)
class MaterialManifestEntry:
    """One authoritative material choice from ``materials.yaml``."""

    name: str
    description: str
    binding_path: str
    tags: tuple[str, ...]

    def as_palette_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "material_path": self.binding_path,
            "description": self.description,
            "tags": list(self.tags),
            "manifest_semantics": material_manifest_semantics(
                self.name, self.description
            ),
        }


@dataclass(frozen=True, slots=True)
class ResolvedMaterialManifest:
    """Validated manifest plus its resolved USD material library."""

    manifest_path: Path
    library_path: Path
    entries: tuple[MaterialManifestEntry, ...]

    @property
    def by_name(self) -> dict[str, MaterialManifestEntry]:
        return {entry.name: entry for entry in self.entries}

    def as_palette(self) -> dict[str, Any]:
        by_tag: dict[str, list[str]] = defaultdict(list)
        materials = []
        for entry in self.entries:
            materials.append(entry.as_palette_record())
            for tag in entry.tags:
                by_tag[tag].append(entry.name)
        return {
            "schema_version": "content-agents.material-palette.v1",
            "materials_yaml": str(self.manifest_path),
            "materials_usd": str(self.library_path),
            "material_count": len(materials),
            "materials": materials,
            "tags": {tag: sorted(names) for tag, names in sorted(by_tag.items())},
        }


def load_material_manifest(
    manifest_path: Path,
    *,
    library_override: Path | None = None,
    validate_material_prims: bool = True,
    allow_empty: bool = False,
) -> ResolvedMaterialManifest:
    """Resolve and validate an authoritative workflow material manifest.

    The YAML must be a mapping with a non-empty ``entries`` list unless a
    syntax-only caller explicitly permits an empty palette. Names and binding paths
    are unique, paths are absolute USD prim paths, and—unless disabled for a narrow
    syntax-only caller—each binding names a real ``UsdShade.Material`` or a
    legacy material-terminal prim that usd-cli can normalize in the resolved
    library.
    """

    path = manifest_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Material manifest does not exist: {path}")
    try:
        with path.open("rb") as stream:
            manifest_bytes = stream.read(MAX_MATERIAL_MANIFEST_BYTES + 1)
        if len(manifest_bytes) > MAX_MATERIAL_MANIFEST_BYTES:
            raise ValueError(
                "Material manifest exceeds the "
                f"{MAX_MATERIAL_MANIFEST_BYTES}-byte limit: {path}"
            )
        raw = yaml.safe_load(manifest_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"Material manifest is not valid UTF-8: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse material manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Material manifest must be a YAML mapping: {path}")

    library_path = _resolve_library_path(path, raw, library_override)
    entries_value = raw.get("entries")
    if not isinstance(entries_value, list) or (not entries_value and not allow_empty):
        raise ValueError(f"Material manifest entries must be a non-empty list: {path}")
    if len(entries_value) > MAX_MATERIAL_MANIFEST_ENTRIES:
        raise ValueError(
            "Material manifest exceeds the "
            f"{MAX_MATERIAL_MANIFEST_ENTRIES}-entry limit: {path}"
        )

    entries: list[MaterialManifestEntry] = []
    names: set[str] = set()
    bindings: set[str] = set()
    for index, value in enumerate(entries_value):
        if not isinstance(value, dict):
            raise ValueError(f"Material manifest entry {index} must be a mapping.")
        name = _required_string(value, "name", index)
        binding = _required_string(value, "binding", index)
        description = _optional_string(value.get("description"))
        if not binding.startswith("/"):
            raise ValueError(
                f"Material manifest entry {index} binding must be an absolute "
                f"USD prim path: {binding!r}"
            )
        if name in names:
            raise ValueError(f"Duplicate material manifest name: {name!r}")
        if binding in bindings:
            raise ValueError(f"Duplicate material manifest binding: {binding!r}")
        names.add(name)
        bindings.add(binding)

        raw_tags = value.get("tags")
        if raw_tags is None:
            tags = tuple(infer_material_tags(name, description))
        elif (
            isinstance(raw_tags, list)
            and len(raw_tags) <= MAX_MATERIAL_MANIFEST_TAGS_PER_ENTRY
            and all(
                isinstance(tag, str)
                and len(tag) <= MAX_MATERIAL_MANIFEST_STRING_CHARS
                and tag.strip()
                for tag in raw_tags
            )
        ):
            tags = tuple(dict.fromkeys(tag.strip() for tag in raw_tags))
        else:
            raise ValueError(
                f"Material manifest entry {index} tags must be a list of "
                "non-empty strings."
            )
        entries.append(
            MaterialManifestEntry(
                name=name,
                description=description,
                binding_path=binding,
                tags=tags,
            )
        )

    if validate_material_prims:
        _validate_library_materials(library_path, entries)
    return ResolvedMaterialManifest(
        manifest_path=path,
        library_path=library_path,
        entries=tuple(entries),
    )


def _resolve_library_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    override: Path | None,
) -> Path:
    if override is not None:
        library_path = override.expanduser()
        if not library_path.is_absolute():
            library_path = manifest_path.parent / library_path
    else:
        value = manifest.get("library_path")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{manifest_path} does not define a non-empty top-level library_path."
            )
        library_path = Path(value.strip()).expanduser()
        if not library_path.is_absolute():
            library_path = manifest_path.parent / library_path
    library_path = library_path.resolve()
    if not library_path.is_file():
        raise ValueError(f"Material USD library does not exist: {library_path}")
    return library_path


def _validate_library_materials(
    library_path: Path, entries: list[MaterialManifestEntry]
) -> None:
    try:
        from pxr import Usd
        from usd_core.materials import is_library_material_prim
    except ImportError as exc:  # pragma: no cover - package requires USD at runtime
        raise RuntimeError(
            "USD Python bindings are required to validate a material manifest."
        ) from exc

    stage = Usd.Stage.Open(str(library_path))
    if stage is None:
        raise ValueError(f"Could not open material USD library: {library_path}")
    invalid: list[str] = []
    for entry in entries:
        prim = stage.GetPrimAtPath(entry.binding_path)
        if not is_library_material_prim(prim):
            invalid.append(f"{entry.name!r} -> {entry.binding_path}")
    if invalid:
        raise ValueError(
            "Material manifest bindings must name UsdShade.Material prims or "
            "legacy prims with authored material terminals in "
            f"{library_path}: {', '.join(invalid)}"
        )


def _required_string(value: dict[str, Any], key: str, index: int) -> str:
    result = _optional_string(value.get(key))
    if not result:
        raise ValueError(
            f"Material manifest entry {index} requires a non-empty {key!r}."
        )
    return result


def _optional_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    if len(value) > MAX_MATERIAL_MANIFEST_STRING_CHARS:
        raise ValueError(
            "Material manifest string exceeds the "
            f"{MAX_MATERIAL_MANIFEST_STRING_CHARS}-character limit."
        )
    return value.strip()


def infer_material_tags(name: str, description: str) -> list[str]:
    """Return deterministic fallback tags when a manifest does not author them."""

    text = f"{name} {description}".lower()
    result = []
    for tag, needles in {
        "black": ("black", "dark"),
        "blue": ("blue", "cyan"),
        "red": ("red", "ruby"),
        "orange": ("orange",),
        "yellow": ("yellow", "gold"),
        "white": ("white", "ivory"),
        "gray": ("gray", "grey", "silver", "gunmetal"),
        "metal": (
            "metal",
            "steel",
            "aluminum",
            "brass",
            "bronze",
            "copper",
            "iron",
            "gold",
            "silver",
        ),
        "plastic": ("plastic",),
        "rubber": ("rubber", "silicone"),
        "glass": ("glass", "transparent", "clear", "translucent"),
        "paint": ("paint", "automotive"),
        "matte": ("matte", "dull", "rough"),
        "glossy": ("gloss", "polished", "reflective", "mirror"),
        "brushed": ("brushed", "grain", "streak"),
    }.items():
        if any(needle in text for needle in needles):
            result.append(tag)
    return result


def material_manifest_semantics(name: str, description: str) -> dict[str, list[str]]:
    """Expose compact semantic hints without changing authoritative manifest data."""

    text = f"{name} {description}".lower()
    groups = {
        "colors": {
            "black": ("black", "charcoal", "jet-black"),
            "blue": ("blue", "navy", "cyan", "turquoise"),
            "brown": ("brown", "russet"),
            "clear": ("clear", "transparent", "colorless"),
            "gold": ("gold", "yellow-gold"),
            "gray": ("gray", "grey", "silver", "gunmetal"),
            "green": ("green",),
            "orange": ("orange",),
            "red": ("red", "ruby"),
            "white": ("white", "ivory"),
            "yellow": ("yellow",),
        },
        "substances": {
            "automotive_paint": ("automotive paint", "car paint"),
            "glass": ("glass",),
            "metal": ("metal", "metallic"),
            "plastic": ("plastic",),
            "rubber": ("rubber",),
            "silicone": ("silicone",),
            "steel": ("steel",),
        },
        "finishes": {
            "brushed": ("brushed", "grain", "streak"),
            "glossy": ("gloss", "reflective", "polished"),
            "matte": ("matte", "dull", "non-glossy", "rough"),
            "painted": ("paint", "paint-coated", "automotive"),
            "polished": ("polished", "mirror"),
        },
    }
    return {
        group: [
            term
            for term, needles in terms.items()
            if any(needle in text for needle in needles)
        ]
        for group, terms in groups.items()
    }
