# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation helpers for generated material library packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ValidationResult:
    """Validation result for a generated material library package."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _asset_attr_paths(stage) -> list[str]:
    from pxr import Sdf

    paths: list[str] = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if value is None:
                continue
            raw_path = value.path if hasattr(value, "path") else str(value)
            if raw_path:
                paths.append(raw_path)
    return paths


def _is_nonblank_png(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return False
            _alpha_min, alpha_max = image.convert("RGBA").getchannel("A").getextrema()
            return alpha_max > 0
    except (OSError, UnidentifiedImageError):
        return False


def validate_generated_material_library(
    materials_manifest_path: str | Path,
) -> ValidationResult:
    """Validate `materials.yaml`, referenced USD material prims, and texture files."""
    from pxr import Usd

    manifest_path = Path(materials_manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        return ValidationResult(
            errors=(f"materials manifest not found: {manifest_path}",)
        )

    try:
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = yaml.safe_load(stream) or {}
    except yaml.YAMLError as exc:
        errors.append(f"failed to parse materials manifest YAML: {exc}")
        return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))

    if not isinstance(manifest, dict):
        errors.append("materials manifest root must be a mapping")
        return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))

    library_raw = manifest.get("library_path")
    if not library_raw:
        errors.append("materials manifest is missing library_path")
        return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))

    library_path = Path(library_raw)
    if not library_path.is_absolute():
        library_path = (manifest_path.parent / library_path).resolve()
    if not library_path.exists():
        errors.append(f"material library USD not found: {library_path}")
        return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))

    stage = Usd.Stage.Open(str(library_path))
    if stage is None:
        errors.append(f"failed to open material library USD: {library_path}")
        return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))

    entries = manifest.get("entries") or []
    if not entries:
        errors.append("materials manifest has no entries")

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"material entry {index} must be a mapping")
            continue
        name = entry.get("name") or f"<entry {index}>"
        binding = entry.get("binding")
        if not binding:
            errors.append(f"material entry {name!r} is missing binding")
            continue
        if not stage.GetPrimAtPath(binding).IsValid():
            errors.append(f"material binding for {name!r} not found: {binding}")

    texture_count = 0
    library_dir = library_path.parent
    for asset_path in _asset_attr_paths(stage):
        if "://" in asset_path or asset_path.startswith("/"):
            warnings.append(
                f"non-relative asset path in material library: {asset_path}"
            )
            continue
        resolved = (library_dir / asset_path).resolve()
        try:
            resolved.relative_to(library_dir.resolve())
        except ValueError:
            errors.append(f"asset path escapes material library package: {asset_path}")
            continue
        texture_count += 1
        if not _is_nonblank_png(resolved):
            errors.append(f"texture is missing, blank, or not PNG: {resolved}")

    return ValidationResult(
        errors=tuple(errors),
        warnings=tuple(warnings),
        metadata={
            "entry_count": len(entries),
            "texture_count": texture_count,
            "library_path": str(library_path),
        },
    )
