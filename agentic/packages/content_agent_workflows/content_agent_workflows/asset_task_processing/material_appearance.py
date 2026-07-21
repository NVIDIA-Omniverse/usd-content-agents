# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendered material-library appearance evidence for Workflow 2."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from content_workbench_agent_client.client import (
    apply_command,
    close_session,
    create_session,
    download_render_artifacts,
    render,
    wait_until_healthy,
)
from PIL import Image, ImageStat, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
    resolve_artifact_path,
)

MATERIAL_APPEARANCE_INDEX_SCHEMA_VERSION = (
    "content-agent-workflows.material-appearance-index.v1"
)
DISPLAY_COLOR_MATCH_SCHEMA_VERSION = (
    "content-agent-workflows.display-color-material-matches.v1"
)
DISPLAY_COLOR_SWATCH_SCHEMA_VERSION = "content-agent-workflows.display-color-swatch.v1"

_SWATCH_RENDER_CONFIG: dict[str, object] = {
    "width": 256,
    "height": 256,
    "direction": "oblique",
    "camera_mode": "frame_sphere",
    "render_quality": "inspection",
    "representative_crop_fraction": 0.2,
}


class MaterialAppearanceError(RuntimeError):
    """Raised when rendered material appearance evidence cannot be measured."""


class RenderedAppearance(BaseModel):
    """Compact measured appearance from one neutral swatch render."""

    model_config = ConfigDict(extra="forbid")

    swatch_path: str
    representative_srgb: list[float] = Field(min_length=3, max_length=3)
    representative_lab: list[float] = Field(min_length=3, max_length=3)


class MaterialAppearanceEntry(RenderedAppearance):
    """Rendered appearance evidence for one material-library entry."""

    material_name: str
    material_path: str
    description: str = ""


class MaterialAppearanceIndex(BaseModel):
    """Cached rendered appearance index for one immutable material library."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MATERIAL_APPEARANCE_INDEX_SCHEMA_VERSION] = (
        MATERIAL_APPEARANCE_INDEX_SCHEMA_VERSION
    )
    cache_key: str
    material_library_yaml: str
    material_library_path: str
    material_library_yaml_digest: str
    material_library_usd_digest: str
    swatch_template_path: str
    swatch_template_digest: str
    render_config: dict[str, object]
    materials: list[MaterialAppearanceEntry]

    @model_validator(mode="after")
    def validate_unique_materials(self) -> MaterialAppearanceIndex:
        names = [material.material_name for material in self.materials]
        if len(names) != len(set(names)):
            raise ValueError("material appearance names must be unique")
        return self


class RankedMaterialAppearance(BaseModel):
    """One perceptually ranked material candidate."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    material_name: str
    material_path: str
    description: str = ""
    delta_e_76: float = Field(ge=0.0)
    swatch_path: str
    representative_srgb: list[float] = Field(min_length=3, max_length=3)
    representative_lab: list[float] = Field(min_length=3, max_length=3)


class DisplayColorCandidateMatch(BaseModel):
    """Nearest rendered materials for one surveyed source candidate."""

    model_config = ConfigDict(extra="forbid")

    prim_path: str
    display_color: list[float] = Field(min_length=3, max_length=3)
    target_swatch_path: str
    target_representative_srgb: list[float] = Field(min_length=3, max_length=3)
    target_representative_lab: list[float] = Field(min_length=3, max_length=3)
    nearest_materials: list[RankedMaterialAppearance]


class DisplayColorMaterialMatches(BaseModel):
    """Agent evidence for prompt-scoped display-color material retrieval."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[DISPLAY_COLOR_MATCH_SCHEMA_VERSION] = (
        DISPLAY_COLOR_MATCH_SCHEMA_VERSION
    )
    work_item_id: str
    task_request_path: str
    task_request_digest: str = Field(min_length=1)
    survey_path: str
    appearance_index_path: str
    scope_paths: list[str]
    target_swatch_schema_version: Literal[DISPLAY_COLOR_SWATCH_SCHEMA_VERSION] = (
        DISPLAY_COLOR_SWATCH_SCHEMA_VERSION
    )
    render_config: dict[str, object]
    top_k: int = Field(ge=1)
    matches: list[DisplayColorCandidateMatch]
    candidates_without_display_color: list[str] = Field(default_factory=list)


def _combined_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "material"


def _clamp_color(color: list[float] | tuple[float, ...]) -> list[float]:
    if len(color) != 3:
        raise ValueError("RGB colors must contain exactly three values")
    result = []
    for component in color:
        numeric = float(component)
        if not math.isfinite(numeric):
            raise ValueError("RGB colors must contain finite values")
        result.append(min(max(numeric, 0.0), 1.0))
    return result


def _color_key(color: list[float] | tuple[float, ...]) -> str:
    return ",".join(f"{component:.6f}" for component in _clamp_color(color))


def _target_swatch_stem(*, color_key: str, render_key: str) -> str:
    color_digest = hashlib.sha256(color_key.encode()).hexdigest()[:12]
    return f"display_color_{color_digest}_{render_key[:12]}"


def srgb_to_lab(color: list[float] | tuple[float, ...]) -> list[float]:
    """Convert normalized display sRGB to CIE Lab using a D65 white point."""

    srgb = _clamp_color(color)

    def linearize(component: float) -> float:
        if component <= 0.04045:
            return component / 12.92
        return ((component + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(component) for component in srgb)
    x = red * 0.4124564 + green * 0.3575761 + blue * 0.1804375
    y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
    z = red * 0.0193339 + green * 0.1191920 + blue * 0.9503041
    x /= 0.95047
    z /= 1.08883

    def lab_component(component: float) -> float:
        delta = 6.0 / 29.0
        if component > delta**3:
            return component ** (1.0 / 3.0)
        return component / (3.0 * delta**2) + 4.0 / 29.0

    fx = lab_component(x)
    fy = lab_component(y)
    fz = lab_component(z)
    return [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]


def delta_e_76(first: list[float], second: list[float]) -> float:
    """Return the CIE76 distance between two Lab colors."""

    if len(first) != 3 or len(second) != 3:
        raise ValueError("Lab colors must contain exactly three values")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))


def representative_srgb(
    image_path: str | Path,
    *,
    crop_fraction: float = 0.2,
) -> list[float]:
    """Measure the median color in the stable center patch of a swatch."""

    if crop_fraction <= 0.0 or crop_fraction > 1.0:
        raise ValueError("crop_fraction must be in (0, 1]")
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            crop_width = max(1, round(width * crop_fraction))
            crop_height = max(1, round(height * crop_fraction))
            left = (width - crop_width) // 2
            top = (height - crop_height) // 2
            crop = rgb.crop((left, top, left + crop_width, top + crop_height))
            median = ImageStat.Stat(crop).median
    except (OSError, UnidentifiedImageError) as exc:
        raise MaterialAppearanceError(
            f"Could not measure rendered swatch image: {image_path}"
        ) from exc
    return [float(component) / 255.0 for component in median[:3]]


def _render_override(
    *,
    workbench_url: str,
    session_id: str,
    material: dict[str, object],
    output_dir: Path,
    name: str,
) -> RenderedAppearance:
    apply_command(
        workbench_url,
        session_id,
        "material_override",
        {
            "prim_path": "/Root/Sphere",
            "space": "source",
            "unbind_existing": True,
            "material": material,
        },
    )
    response = render(
        workbench_url,
        session_id,
        {
            "width": int(_SWATCH_RENDER_CONFIG["width"]),
            "height": int(_SWATCH_RENDER_CONFIG["height"]),
            "use_session_camera": True,
            "render_quality": str(_SWATCH_RENDER_CONFIG["render_quality"]),
            "save_camera_json": True,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    record = download_render_artifacts(
        workbench_url=workbench_url,
        response=response,
        image_path=output_dir / f"{name}.png",
        response_path=output_dir / f"{name}_response.json",
        camera_path=output_dir / f"{name}_camera.json",
    )
    image_path = Path(str(record["image_path"])).expanduser().resolve()
    measured = representative_srgb(
        image_path,
        crop_fraction=float(_SWATCH_RENDER_CONFIG["representative_crop_fraction"]),
    )
    return RenderedAppearance(
        swatch_path=str(image_path),
        representative_srgb=measured,
        representative_lab=srgb_to_lab(measured),
    )


def _material_manifest_entries(
    yaml_path: Path,
    expected_library_path: Path,
) -> list[dict[str, str]]:
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read material library {yaml_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError(f"Invalid material library metadata: {yaml_path}")
    configured_library = payload.get("library_path")
    if isinstance(configured_library, str):
        resolved = resolve_artifact_path(configured_library, base_dir=yaml_path.parent)
        if resolved != expected_library_path:
            raise ValueError("Material YAML and USD library paths differ")
    entries: list[dict[str, str]] = []
    for raw in payload["entries"]:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        binding = raw.get("binding")
        if not isinstance(name, str) or not isinstance(binding, str):
            continue
        entries.append(
            {
                "name": name,
                "binding": binding,
                "description": str(raw.get("description") or ""),
            }
        )
    if not entries:
        raise ValueError(f"Material library has no usable entries: {yaml_path}")
    return entries


def build_material_appearance_index(
    *,
    material_library_yaml: str | Path,
    material_library_path: str | Path,
    swatch_template_path: str | Path,
    cache_dir: str | Path,
    workbench_url: str,
) -> tuple[MaterialAppearanceIndex, Path]:
    """Render and cache neutral swatches for every library material."""

    yaml_path = Path(material_library_yaml).expanduser().resolve()
    library_path = Path(material_library_path).expanduser().resolve()
    template_path = Path(swatch_template_path).expanduser().resolve()
    for label, path in (
        ("material YAML", yaml_path),
        ("material USD", library_path),
        ("swatch template", template_path),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    manifest_entries = _material_manifest_entries(yaml_path, library_path)
    yaml_digest = file_sha256(yaml_path)
    usd_digest = file_sha256(library_path)
    template_digest = file_sha256(template_path)
    cache_key = _combined_digest(
        {
            "schema_version": MATERIAL_APPEARANCE_INDEX_SCHEMA_VERSION,
            "material_library_yaml_digest": yaml_digest,
            "material_library_usd_digest": usd_digest,
            "swatch_template_digest": template_digest,
            "render_config": _SWATCH_RENDER_CONFIG,
        }
    )
    index_dir = Path(cache_dir).expanduser().resolve() / cache_key
    index_path = index_dir / "material_appearance_index.json"
    if index_path.is_file():
        try:
            existing = MaterialAppearanceIndex.model_validate(load_json(index_path))
        except (OSError, ValueError):
            existing = None
        if (
            existing is not None
            and existing.cache_key == cache_key
            and all(Path(item.swatch_path).is_file() for item in existing.materials)
        ):
            return existing, index_path

    swatches_dir = index_dir / "swatches"
    swatches_dir.mkdir(parents=True, exist_ok=True)
    wait_until_healthy(workbench_url, timeout_seconds=30.0)
    session = create_session(
        workbench_url,
        {
            "scene_path": str(template_path),
            "optimize": False,
            "clear_materials": True,
            "width": int(_SWATCH_RENDER_CONFIG["width"]),
            "height": int(_SWATCH_RENDER_CONFIG["height"]),
        },
    )
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("Workbench did not return a swatch session_id")
    materials: list[MaterialAppearanceEntry] = []
    try:
        apply_command(
            workbench_url,
            session_id,
            "frame",
            {
                "prim_path": "/Root/Sphere",
                "direction": _SWATCH_RENDER_CONFIG["direction"],
                "margin": 1.25,
            },
        )
        for index, entry in enumerate(manifest_entries):
            stem = f"{index:03d}_{_safe_name(entry['name'])}"
            image_path = swatches_dir / f"{stem}.png"
            if image_path.is_file():
                measured = representative_srgb(
                    image_path,
                    crop_fraction=float(
                        _SWATCH_RENDER_CONFIG["representative_crop_fraction"]
                    ),
                )
                appearance = RenderedAppearance(
                    swatch_path=str(image_path),
                    representative_srgb=measured,
                    representative_lab=srgb_to_lab(measured),
                )
            else:
                appearance = _render_override(
                    workbench_url=workbench_url,
                    session_id=session_id,
                    material={
                        "source": "material_library",
                        "library_path": str(library_path),
                        "material_name": entry["name"],
                        "material_path": entry["binding"],
                    },
                    output_dir=swatches_dir,
                    name=stem,
                )
            materials.append(
                MaterialAppearanceEntry(
                    **appearance.model_dump(),
                    material_name=entry["name"],
                    material_path=entry["binding"],
                    description=entry["description"],
                )
            )
    finally:
        close_session(workbench_url, session_id)

    appearance_index = MaterialAppearanceIndex(
        cache_key=cache_key,
        material_library_yaml=str(yaml_path),
        material_library_path=str(library_path),
        material_library_yaml_digest=yaml_digest,
        material_library_usd_digest=usd_digest,
        swatch_template_path=str(template_path),
        swatch_template_digest=template_digest,
        render_config=dict(_SWATCH_RENDER_CONFIG),
        materials=materials,
    )
    atomic_write_json(index_path, appearance_index)
    return appearance_index, index_path


def render_display_color_targets(
    *,
    colors: list[list[float]],
    swatch_template_path: str | Path,
    output_dir: str | Path,
    workbench_url: str,
) -> dict[str, RenderedAppearance]:
    """Render each unique source display color through the swatch pipeline."""

    template_path = Path(swatch_template_path).expanduser().resolve()
    if not template_path.is_file():
        raise ValueError(f"Swatch template does not exist: {template_path}")
    render_key = _combined_digest(
        {
            "schema_version": DISPLAY_COLOR_SWATCH_SCHEMA_VERSION,
            "swatch_template_digest": file_sha256(template_path),
            "render_config": _SWATCH_RENDER_CONFIG,
        }
    )
    unique = {_color_key(color): _clamp_color(color) for color in colors}
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    appearances: dict[str, RenderedAppearance] = {}
    missing: dict[str, list[float]] = {}
    for key, color in unique.items():
        stem = _target_swatch_stem(color_key=key, render_key=render_key)
        image_path = target_dir / f"{stem}.png"
        if image_path.is_file():
            measured = representative_srgb(
                image_path,
                crop_fraction=float(
                    _SWATCH_RENDER_CONFIG["representative_crop_fraction"]
                ),
            )
            appearances[key] = RenderedAppearance(
                swatch_path=str(image_path),
                representative_srgb=measured,
                representative_lab=srgb_to_lab(measured),
            )
        else:
            missing[key] = color
    if not missing:
        return appearances

    wait_until_healthy(workbench_url, timeout_seconds=30.0)
    session = create_session(
        workbench_url,
        {
            "scene_path": str(template_path),
            "optimize": False,
            "clear_materials": True,
            "width": int(_SWATCH_RENDER_CONFIG["width"]),
            "height": int(_SWATCH_RENDER_CONFIG["height"]),
        },
    )
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("Workbench did not return a target swatch session_id")
    try:
        apply_command(
            workbench_url,
            session_id,
            "frame",
            {
                "prim_path": "/Root/Sphere",
                "direction": _SWATCH_RENDER_CONFIG["direction"],
                "margin": 1.25,
            },
        )
        for key, color in missing.items():
            stem = _target_swatch_stem(color_key=key, render_key=render_key)
            appearances[key] = _render_override(
                workbench_url=workbench_url,
                session_id=session_id,
                material={
                    "display_name": "Authored Display Color",
                    "color": color,
                    "roughness": 0.45,
                    "metallic": 0.0,
                },
                output_dir=target_dir,
                name=stem,
            )
    finally:
        close_session(workbench_url, session_id)
    return appearances


def _path_is_scoped(path: str, scope_paths: list[str]) -> bool:
    return any(
        path == scope or path.startswith(scope.rstrip("/") + "/")
        for scope in scope_paths
    )


def rank_display_color_candidates(
    *,
    work_item_id: str,
    task_request_path: str | Path,
    task_request_digest: str,
    survey_path: str | Path,
    survey: dict[str, Any],
    appearance_index_path: str | Path,
    appearance_index: MaterialAppearanceIndex,
    target_appearances: dict[str, RenderedAppearance],
    scope_paths: list[str],
    top_k: int,
) -> DisplayColorMaterialMatches:
    """Rank rendered library materials without choosing or applying one."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    normalized_scopes = list(
        dict.fromkeys(scope.rstrip("/") or "/" for scope in scope_paths)
    )
    if not normalized_scopes or any(
        not scope.startswith("/") for scope in normalized_scopes
    ):
        raise ValueError("scope_paths must contain absolute USD prim paths")
    materials = appearance_index.materials
    if not materials:
        raise ValueError("material appearance index is empty")
    matches: list[DisplayColorCandidateMatch] = []
    missing: list[str] = []
    candidates = survey.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("material survey candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        prim_path = candidate.get("prim_path")
        if not isinstance(prim_path, str) or not _path_is_scoped(
            prim_path, normalized_scopes
        ):
            continue
        display_color = candidate.get("display_color")
        if not isinstance(display_color, list | tuple) or len(display_color) != 3:
            missing.append(prim_path)
            continue
        normalized_color = _clamp_color(display_color)
        target = target_appearances.get(_color_key(normalized_color))
        if target is None:
            raise ValueError(
                f"Missing rendered target appearance for {normalized_color}"
            )
        ranked = sorted(
            (
                (
                    delta_e_76(target.representative_lab, material.representative_lab),
                    material,
                )
                for material in materials
            ),
            key=lambda item: (item[0], item[1].material_name),
        )[: min(top_k, len(materials))]
        nearest = [
            RankedMaterialAppearance(
                rank=index,
                material_name=material.material_name,
                material_path=material.material_path,
                description=material.description,
                delta_e_76=distance,
                swatch_path=material.swatch_path,
                representative_srgb=material.representative_srgb,
                representative_lab=material.representative_lab,
            )
            for index, (distance, material) in enumerate(ranked, start=1)
        ]
        matches.append(
            DisplayColorCandidateMatch(
                prim_path=prim_path,
                display_color=normalized_color,
                target_swatch_path=target.swatch_path,
                target_representative_srgb=target.representative_srgb,
                target_representative_lab=target.representative_lab,
                nearest_materials=nearest,
            )
        )
    if not matches:
        raise ValueError("No display-color candidates matched the requested scopes")
    return DisplayColorMaterialMatches(
        work_item_id=work_item_id,
        task_request_path=str(Path(task_request_path).expanduser().resolve()),
        task_request_digest=task_request_digest,
        survey_path=str(Path(survey_path).expanduser().resolve()),
        appearance_index_path=str(Path(appearance_index_path).expanduser().resolve()),
        scope_paths=normalized_scopes,
        render_config=dict(_SWATCH_RENDER_CONFIG),
        top_k=top_k,
        matches=matches,
        candidates_without_display_color=sorted(missing),
    )
