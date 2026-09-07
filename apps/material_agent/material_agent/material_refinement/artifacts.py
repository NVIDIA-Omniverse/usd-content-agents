# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation, identity, and portable retargeting for material map artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from material_agent.material_library_generation.schema import TextureMapSet


def artifact_sha256(path: Path) -> str:
    """Return the byte identity of one local artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class MaterialMapValidation:
    """Auditable validation result for one albedo/normal/ORM set."""

    width: int
    height: int
    sha256: dict[str, str]
    byte_size: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "sha256": dict(self.sha256),
            "byte_size": dict(self.byte_size),
        }


def _load_rgb(path: Path, *, channel: str) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"material {channel} map does not exist")
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
    if rgb.width <= 0 or rgb.height <= 0:
        raise ValueError(f"material {channel} map has invalid dimensions")
    return rgb


def validate_material_maps(textures: TextureMapSet) -> MaterialMapValidation:
    """Require three distinct, readable, dimensionally consistent RGB maps."""

    paths = {
        "albedo": textures.albedo.resolve(),
        "normal": textures.normal.resolve(),
        "orm": textures.orm.resolve(),
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError("material maps must use distinct artifact paths")
    images = {
        channel: _load_rgb(path, channel=channel) for channel, path in paths.items()
    }
    dimensions = {(image.width, image.height) for image in images.values()}
    if len(dimensions) != 1:
        raise ValueError("material maps must have matching dimensions")
    width, height = dimensions.pop()
    return MaterialMapValidation(
        width=width,
        height=height,
        sha256={channel: artifact_sha256(path) for channel, path in paths.items()},
        byte_size={channel: path.stat().st_size for channel, path in paths.items()},
    )


def materialize_candidate_maps(
    source: TextureMapSet,
    *,
    output_dir: Path,
) -> tuple[TextureMapSet, MaterialMapValidation, MaterialMapValidation]:
    """Create portable RGB copies without changing generated material evidence."""

    source_validation = validate_material_maps(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = TextureMapSet(
        albedo=output_dir / "albedo.png",
        normal=output_dir / "normal.png",
        orm=output_dir / "orm.png",
    )

    with Image.open(source.albedo) as image:
        image.convert("RGB").save(output.albedo)
    with Image.open(source.normal) as image:
        image.convert("RGB").save(output.normal)
    with Image.open(source.orm) as image:
        image.convert("RGB").save(output.orm)

    output_validation = validate_material_maps(output)
    return output, source_validation, output_validation


__all__ = [
    "MaterialMapValidation",
    "artifact_sha256",
    "materialize_candidate_maps",
    "validate_material_maps",
]
