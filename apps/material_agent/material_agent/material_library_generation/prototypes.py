# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prototype material selection for generated material libraries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from material_agent.material_library_generation.schema import MaterialRecipe

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "and",
    "for",
    "of",
    "or",
    "the",
    "to",
    "used",
    "with",
}
_TRANSLUCENT_TOKENS = {
    "clear",
    "cloudy",
    "frosted",
    "transparent",
    "translucent",
    "milky",
}
_OPTICAL_TOKENS = _TRANSLUCENT_TOKENS | {"acrylic", "glass"}
_COLOR_TOKENS = {
    "beige",
    "black",
    "blue",
    "brown",
    "bronze",
    "copper",
    "cyan",
    "gold",
    "gray",
    "green",
    "grey",
    "gunmetal",
    "ivory",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "turquoise",
    "white",
    "yellow",
}
_COLOR_EQUIVALENTS = {
    "white": {"beige", "cream", "ivory", "white"},
    "gray": {"black", "gray", "grey", "gunmetal", "silver"},
    "grey": {"black", "gray", "grey", "gunmetal", "silver"},
    "silver": {"aluminum", "gray", "grey", "silver", "steel"},
}
_METAL_TOKENS = {
    "aluminum",
    "brass",
    "bronze",
    "copper",
    "iron",
    "metal",
    "metallic",
    "silver",
    "steel",
}
_SPECIFIC_SUBSTANCE_TOKENS = {
    "aluminum",
    "acrylic",
    "brass",
    "bronze",
    "copper",
    "glass",
    "iron",
    "plastic",
    "rubber",
    "silicone",
    "steel",
}
_GLOSS_TOKENS = {"clearcoat", "gloss", "glossy", "polished", "reflective", "shine"}
_SMOOTH_TOKENS = {"clean", "even", "satin", "smooth", "soft"}
_SHELL_TOKENS = {"body", "casing", "enclosure", "housing", "lid", "shell"}


@dataclass(frozen=True)
class MaterialPrototype:
    """A default-library material candidate used as an authoring prototype."""

    name: str
    description: str
    binding: str
    library_path: Path
    base_color: tuple[float, float, float] | None = None
    metalness: float | None = None

    @property
    def text(self) -> str:
        return f"{self.name} {self.description}".strip()

    def to_source_dict(self, *, score: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "binding": self.binding,
            "library_path": str(self.library_path),
            "score": float(score),
        }


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        token for token in _TOKEN_RE.findall(value.lower()) if token not in _STOPWORDS
    }


def _recipe_text(recipe: MaterialRecipe) -> str:
    parts = [
        recipe.name,
        recipe.description,
        recipe.appearance_prompt,
        recipe.color or "",
        recipe.material or "",
        recipe.finish or "",
    ]
    parts.extend(part.semantic_label for part in recipe.intended_parts)
    parts.extend(part.evidence for part in recipe.intended_parts)
    return " ".join(parts)


def _as_color(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        channels = tuple(float(channel) for channel in value)
    except TypeError:
        return None
    if len(channels) != 3:
        return None
    return channels


def _read_material_input(stage: Any, binding: str, name: str) -> Any:
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(binding)
    if not prim or not prim.IsValid():
        return None
    material = UsdShade.Material(prim)
    material_input = material.GetInput(name)
    if material_input:
        return material_input.Get()

    for child in prim.GetChildren():
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        for input_name in (name, "diffuseColor" if name == "base_color" else name):
            shader_input = shader.GetInput(input_name)
            if shader_input:
                value = shader_input.Get()
                if value is not None:
                    return value
    return None


def _manifest_library_path(data: dict[str, Any], manifest_path: Path) -> Path | None:
    raw_path = data.get("library_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    return path


def load_material_prototypes_from_manifest(
    manifest_path: str | Path,
) -> tuple[MaterialPrototype, ...]:
    """Load prototype candidates from a canonical Material Agent materials.yaml."""

    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    library_path = _manifest_library_path(data, manifest_path)
    if library_path is None:
        return ()
    return load_material_prototypes_from_data(data, base_dir=manifest_path.parent)


def load_material_prototypes_from_data(
    data: dict[str, Any] | None,
    *,
    base_dir: str | Path | None = None,
) -> tuple[MaterialPrototype, ...]:
    """Load prototype candidates from inline materials data."""

    if not isinstance(data, dict):
        return ()

    raw_library_path = data.get("library_path")
    if not raw_library_path:
        return ()
    library_path = Path(str(raw_library_path))
    if not library_path.is_absolute() and base_dir is not None:
        library_path = (Path(base_dir) / library_path).resolve()
    else:
        library_path = library_path.resolve()

    entries = data.get("entries") or []
    if not entries:
        return ()

    stage = None
    if library_path.exists():
        from pxr import Usd

        stage = Usd.Stage.Open(str(library_path))

    prototypes: list[MaterialPrototype] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        binding = str(entry.get("binding") or "").strip()
        if not name or not binding:
            continue
        base_color = None
        metalness = None
        if stage is not None:
            base_color = _as_color(_read_material_input(stage, binding, "base_color"))
            raw_metalness = _read_material_input(stage, binding, "base_metalness")
            try:
                metalness = float(raw_metalness) if raw_metalness is not None else None
            except (TypeError, ValueError):
                metalness = None
        prototypes.append(
            MaterialPrototype(
                name=name,
                description=str(entry.get("description") or ""),
                binding=binding,
                library_path=library_path,
                base_color=base_color,
                metalness=metalness,
            )
        )
    return tuple(prototypes)


def _color_similarity(
    recipe_color: tuple[float, float, float],
    prototype_color: tuple[float, float, float] | None,
) -> float:
    if prototype_color is None:
        return 0.0
    distance = math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(recipe_color, prototype_color, strict=False)
        )
    )
    return max(0.0, 1.0 - distance / math.sqrt(3.0))


def _color_tokens_match(
    recipe_tokens: set[str],
    candidate_tokens: set[str],
) -> bool:
    recipe_colors = recipe_tokens & _COLOR_TOKENS
    candidate_colors = candidate_tokens & _COLOR_TOKENS
    if not recipe_colors or not candidate_colors:
        return True
    expanded = set(recipe_colors)
    for color in recipe_colors:
        expanded.update(_COLOR_EQUIVALENTS.get(color, ()))
    return bool(expanded & candidate_colors)


def _is_metal_recipe(recipe_tokens: set[str], recipe: MaterialRecipe) -> bool:
    if recipe.pbr_hints.metallic >= 0.5:
        return True
    return bool(recipe_tokens & _METAL_TOKENS)


def _is_optical_recipe(recipe_tokens: set[str], recipe: MaterialRecipe) -> bool:
    if recipe.pbr_hints.transmission > 0.0 or recipe.pbr_hints.opacity < 1.0:
        return True
    return bool(recipe_tokens & _OPTICAL_TOKENS)


def score_material_prototype(
    recipe: MaterialRecipe,
    prototype: MaterialPrototype,
) -> float:
    """Score how useful a default material is as a prototype for a recipe."""

    recipe_tokens = _tokens(_recipe_text(recipe))
    candidate_tokens = _tokens(prototype.text)
    if not recipe_tokens or not candidate_tokens:
        return 0.0

    overlap = len(recipe_tokens & candidate_tokens) / len(
        recipe_tokens | candidate_tokens
    )
    score = overlap

    score += 3.0 * _color_similarity(recipe.base_color_hint, prototype.base_color)

    if recipe.material:
        material_tokens = _tokens(recipe.material)
        material_overlap = material_tokens & candidate_tokens
        score += 0.35 * len(material_overlap)

    recipe_substances = recipe_tokens & _SPECIFIC_SUBSTANCE_TOKENS
    candidate_substances = candidate_tokens & _SPECIFIC_SUBSTANCE_TOKENS
    if recipe_substances:
        substance_overlap = recipe_substances & candidate_substances
        score += 0.8 * len(substance_overlap)
        if candidate_substances and not substance_overlap:
            score -= 0.6

    if recipe.finish:
        finish_tokens = _tokens(recipe.finish)
        score += 0.35 * len(finish_tokens & candidate_tokens)
        if finish_tokens & {"gloss", "glossy", "polished"}:
            score += 0.35 * bool(candidate_tokens & _GLOSS_TOKENS)
        if finish_tokens & {"satin", "smooth"}:
            score += 0.25 * bool(candidate_tokens & (_SMOOTH_TOKENS | _GLOSS_TOKENS))
        if finish_tokens & {"matte", "flat"}:
            score += 0.35 * bool(candidate_tokens & {"matte", "flat", "rough"})

    is_metal = _is_metal_recipe(recipe_tokens, recipe)
    is_optical = _is_optical_recipe(recipe_tokens, recipe)
    candidate_is_optical = bool(candidate_tokens & _OPTICAL_TOKENS)
    if is_optical:
        if candidate_is_optical:
            score += 1.4
        else:
            score -= 1.4
        if candidate_tokens & {"paint", "rubber", "metal", "metallic"}:
            score -= 0.8

    if is_metal:
        if candidate_tokens & _METAL_TOKENS:
            score += 0.8
        if prototype.metalness is not None:
            score += 0.5 * max(
                0.0, 1.0 - abs(recipe.pbr_hints.metallic - prototype.metalness)
            )
    else:
        if prototype.metalness is not None and prototype.metalness <= 0.3:
            score += 0.25
        if candidate_tokens & _METAL_TOKENS:
            score -= 1.2 if "paint" in candidate_tokens else 2.0

    if recipe_tokens & _SHELL_TOKENS and candidate_tokens & {"car", "paint", "painted"}:
        score += 0.45
    if recipe_tokens & _SHELL_TOKENS and {"car", "paint"} <= candidate_tokens:
        score += 0.45

    if candidate_tokens & {"rubber", "silicone"} and not (
        recipe_tokens & {"rubber", "silicone"}
    ):
        score -= 0.65

    if not _color_tokens_match(recipe_tokens, candidate_tokens):
        score -= 1.4

    if candidate_is_optical and not is_optical:
        score -= 2.0

    return score


def select_material_prototype(
    recipe: MaterialRecipe,
    prototypes: tuple[MaterialPrototype, ...] | list[MaterialPrototype],
    *,
    min_score: float = 0.75,
) -> tuple[MaterialPrototype, float] | None:
    """Return the best prototype material for a recipe when the match is credible."""

    best: tuple[MaterialPrototype, float] | None = None
    for prototype in prototypes:
        score = score_material_prototype(recipe, prototype)
        if best is None or score > best[1]:
            best = (prototype, score)
    if best is None or best[1] < min_score:
        return None
    return best
