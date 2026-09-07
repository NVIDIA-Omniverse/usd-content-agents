# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration contracts for rendered material refinement and variation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_understanding.optimization import (
    OPTIMIZER_AUTO,
    OptimizerSettings,
    TunableParam,
)
from world_understanding.rendering_backend_contract import (
    validate_rendering_backend_name,
)

from material_agent.material_library_generation.schema import TextureMapSet
from material_agent.material_profiles import normalize_material_profile

_DEFAULT_OUTPUT_DIR = "material_refinement_output"
_MAX_RENDER_VIEWS = 8
_MIXED_METALNESS_EPSILON = 1.0e-6


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def _reject_unknown_keys(
    values: dict[str, Any], *, allowed: set[str], field_name: str
) -> None:
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown field(s): {', '.join(unknown)}"
        )


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field_name=field_name)


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _unit_float(value: Any, *, field_name: str) -> float:
    number = _finite_float(value, field_name=field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return number


def _positive_float(value: Any, *, field_name: str) -> float:
    number = _finite_float(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _base_color(value: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three normalized channels")
    return (
        _unit_float(value[0], field_name=f"{field_name}[0]"),
        _unit_float(value[1], field_name=f"{field_name}[1]"),
        _unit_float(value[2], field_name=f"{field_name}[2]"),
    )


def _resolve_path(value: Any, *, base_dir: Path, field_name: str) -> Path:
    if isinstance(value, Path):
        path = value.expanduser()
    else:
        path = Path(_non_empty_string(value, field_name=field_name)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class MaterialObjectiveWeights:
    """Internal weights for measurable proxy-objective components."""

    color: float = 0.7
    roughness: float = 0.15
    metallic: float = 0.15

    def __post_init__(self) -> None:
        values = (self.color, self.roughness, self.metallic)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("objective weights must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "color": self.color,
            "roughness": self.roughness,
            "metallic": self.metallic,
        }


@dataclass(frozen=True)
class MaterialRefinementGoal:
    """User-supplied appearance request for material refinement."""

    appearance_prompt: str

    def __post_init__(self) -> None:
        _non_empty_string(self.appearance_prompt, field_name="goal.appearance_prompt")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path,
    ) -> MaterialRefinementGoal:
        _reject_unknown_keys(
            data,
            allowed={"appearance_prompt"},
            field_name="goal",
        )
        return cls(
            appearance_prompt=_non_empty_string(
                data.get("appearance_prompt"),
                field_name="goal.appearance_prompt",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"appearance_prompt": self.appearance_prompt}


@dataclass(frozen=True)
class TextureVariationSettings:
    """Texture Variation API settings shared by every optimizer trial."""

    endpoint: str | None = None
    strength_min: float = 0.2
    strength_max: float = 1.0
    engine: str | None = None
    texture_size: int | None = 256
    max_artifact_bytes: int = 256 * 1024 * 1024
    custom_parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 900.0
    poll_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.endpoint is not None:
            _non_empty_string(self.endpoint, field_name="variation.endpoint")
        minimum = _unit_float(self.strength_min, field_name="variation.strength_min")
        maximum = _unit_float(self.strength_max, field_name="variation.strength_max")
        if minimum > maximum:
            raise ValueError("variation.strength_min cannot exceed strength_max")
        if self.texture_size is not None:
            _positive_int(self.texture_size, field_name="variation.texture_size")
        _positive_int(
            self.max_artifact_bytes,
            field_name="variation.max_artifact_bytes",
        )
        _positive_float(self.timeout_seconds, field_name="variation.timeout_seconds")
        _positive_float(
            self.poll_interval_seconds,
            field_name="variation.poll_interval_seconds",
        )
        object.__setattr__(self, "custom_parameters", dict(self.custom_parameters))

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> TextureVariationSettings:
        values = data or {}
        _reject_unknown_keys(
            values,
            allowed={
                "endpoint",
                "strength_min",
                "strength_max",
                "engine",
                "texture_size",
                "max_artifact_bytes",
                "custom_parameters",
                "timeout_seconds",
                "poll_interval_seconds",
            },
            field_name="variation",
        )
        raw_custom = values.get("custom_parameters", {})
        custom = _mapping(raw_custom, field_name="variation.custom_parameters")
        raw_size = values.get("texture_size", 256)
        return cls(
            endpoint=_optional_string(
                values.get("endpoint"), field_name="variation.endpoint"
            ),
            strength_min=_unit_float(
                values.get("strength_min", 0.2),
                field_name="variation.strength_min",
            ),
            strength_max=_unit_float(
                values.get("strength_max", 1.0),
                field_name="variation.strength_max",
            ),
            engine=_optional_string(
                values.get("engine"), field_name="variation.engine"
            ),
            texture_size=(
                _positive_int(raw_size, field_name="variation.texture_size")
                if raw_size is not None
                else None
            ),
            max_artifact_bytes=_positive_int(
                values.get("max_artifact_bytes", 256 * 1024 * 1024),
                field_name="variation.max_artifact_bytes",
            ),
            custom_parameters=custom,
            timeout_seconds=_positive_float(
                values.get("timeout_seconds", 900.0),
                field_name="variation.timeout_seconds",
            ),
            poll_interval_seconds=_positive_float(
                values.get("poll_interval_seconds", 1.0),
                field_name="variation.poll_interval_seconds",
            ),
        )

    @property
    def search_params(self) -> tuple[TunableParam, ...]:
        """Return standardized Texture Variation controls for shared tuning."""

        return (TunableParam("strength", self.strength_min, self.strength_max),)


@dataclass(frozen=True)
class MaterialRefinementSearchSettings:
    """Source-anchored bounds for portable material controls."""

    max_color_delta: float = 0.25
    max_roughness_delta: float = 0.30
    max_mixed_metallic_delta: float = 0.10

    def __post_init__(self) -> None:
        _unit_float(self.max_color_delta, field_name="search.max_color_delta")
        _unit_float(self.max_roughness_delta, field_name="search.max_roughness_delta")
        mixed_delta = _unit_float(
            self.max_mixed_metallic_delta,
            field_name="search.max_mixed_metallic_delta",
        )
        if mixed_delta > 0.4:
            raise ValueError("search.max_mixed_metallic_delta must not exceed 0.4")

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any] | None
    ) -> MaterialRefinementSearchSettings:
        values = dict(data or {})
        _reject_unknown_keys(
            values,
            allowed={
                "max_color_delta",
                "max_roughness_delta",
                "max_mixed_metallic_delta",
            },
            field_name="search",
        )
        return cls(
            max_color_delta=_unit_float(
                values.get("max_color_delta", 0.25),
                field_name="search.max_color_delta",
            ),
            max_roughness_delta=_unit_float(
                values.get("max_roughness_delta", 0.30),
                field_name="search.max_roughness_delta",
            ),
            max_mixed_metallic_delta=_unit_float(
                values.get("max_mixed_metallic_delta", 0.10),
                field_name="search.max_mixed_metallic_delta",
            ),
        )

    def parameters_for_values(
        self,
        *,
        base_color: tuple[float, float, float],
        roughness: float,
        metallic: float,
        variation: TextureVariationSettings,
        include_texture_variation: bool,
    ) -> tuple[TunableParam, ...]:
        """Build bounded controls anchored to measured source values."""

        resolved_color = _base_color(base_color, field_name="source.base_color")
        resolved_roughness = _unit_float(roughness, field_name="source.roughness")
        resolved_metallic = _unit_float(metallic, field_name="source.metallic")

        color_params = tuple(
            TunableParam(
                f"base_color_{channel}",
                max(0.0, value - self.max_color_delta),
                min(1.0, value + self.max_color_delta),
            )
            for channel, value in zip(("r", "g", "b"), resolved_color, strict=True)
        )
        metallic_min, metallic_max = self.metallic_bounds(resolved_metallic)
        return (
            *color_params,
            TunableParam(
                "roughness",
                max(0.0, resolved_roughness - self.max_roughness_delta),
                min(1.0, resolved_roughness + self.max_roughness_delta),
            ),
            TunableParam("metallic", metallic_min, metallic_max),
            *(variation.search_params if include_texture_variation else ()),
        )

    def metallic_bounds(self, value: float) -> tuple[float, float]:
        """Preserve dielectric, metallic, and mixed metalness classes."""

        if value in {0.0, 1.0}:
            return value, value
        delta = self.max_mixed_metallic_delta
        if value <= 0.1:
            return max(0.0, value - delta), min(0.1, value + delta)
        if value >= 0.9:
            return max(0.9, value - delta), min(1.0, value + delta)
        lower_limit = min(0.1 + _MIXED_METALNESS_EPSILON, value)
        upper_limit = max(0.9 - _MIXED_METALNESS_EPSILON, value)
        return (
            max(lower_limit, value - delta),
            min(upper_limit, value + delta),
        )


@dataclass(frozen=True)
class MaterialRefinementRenderSettings:
    """Existing Material Agent render-task settings for canonical swatches."""

    config: dict[str, Any]

    def __post_init__(self) -> None:
        values = dict(self.config)
        values["backend"] = validate_rendering_backend_name(
            values.get("backend", "ovrtx")
        )
        values["image_width"] = _positive_int(
            values.get("image_width", 768), field_name="render.image_width"
        )
        values["image_height"] = _positive_int(
            values.get("image_height", values["image_width"]),
            field_name="render.image_height",
        )
        values["camera_margin"] = _positive_float(
            values.get("camera_margin", 1.15), field_name="render.camera_margin"
        )
        raw_corners = values.get("camera_corners", ("+x+y+z", "-x+y+z"))
        if isinstance(raw_corners, str):
            raw_corners = (raw_corners,)
        if not isinstance(raw_corners, list | tuple) or not raw_corners:
            raise ValueError("render.camera_corners must be a non-empty list")
        values["camera_corners"] = [
            _non_empty_string(corner, field_name="render.camera_corners")
            for corner in raw_corners
        ]
        if len(values["camera_corners"]) > _MAX_RENDER_VIEWS:
            raise ValueError(
                f"render supports at most {_MAX_RENDER_VIEWS} camera views"
            )
        values["background_color"] = list(
            _base_color(
                values.get("background_color", (0.18, 0.18, 0.18)),
                field_name="render.background_color",
            )
        )
        object.__setattr__(self, "config", values)

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any] | None
    ) -> MaterialRefinementRenderSettings:
        return cls(config=dict(data or {}))

    @property
    def backend(self) -> str:
        return str(self.config["backend"])

    def to_task_config(self) -> dict[str, Any]:
        return dict(self.config)


@dataclass(frozen=True)
class MaterialRefinementJudgeSettings:
    """Material Agent VLM provider and visual-approval settings."""

    vlm: dict[str, Any]
    score_threshold: float = 0.7
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        _unit_float(self.score_threshold, field_name="judge.score_threshold")
        if _finite_float(self.temperature, field_name="judge.temperature") < 0.0:
            raise ValueError("judge.temperature must be non-negative")
        _positive_int(self.max_tokens, field_name="judge.max_tokens")
        object.__setattr__(self, "vlm", dict(self.vlm))

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any] | None
    ) -> MaterialRefinementJudgeSettings:
        values = dict(data or {})
        _reject_unknown_keys(
            values,
            allowed={"vlm", "score_threshold", "temperature", "max_tokens"},
            field_name="judge",
        )
        vlm = _mapping(values.get("vlm", {}), field_name="judge.vlm")
        return cls(
            vlm=vlm,
            score_threshold=_unit_float(
                values.get("score_threshold", 0.7),
                field_name="judge.score_threshold",
            ),
            temperature=_finite_float(
                values.get("temperature", vlm.get("temperature", 0.1)),
                field_name="judge.temperature",
            ),
            max_tokens=_positive_int(
                values.get("max_tokens", vlm.get("max_tokens", 2048)),
                field_name="judge.max_tokens",
            ),
        )

    def provider_evidence(self) -> dict[str, str]:
        evidence: dict[str, str] = {}
        for key in ("backend", "provider", "model"):
            value = self.vlm.get(key)
            if isinstance(value, str) and value.strip():
                evidence[key] = value.strip()
        return evidence


@dataclass(frozen=True)
class MaterialRefinementSource:
    """Existing material state that refinement must preserve and improve."""

    material_id: str | None = None
    name: str = "Source Material"
    description: str | None = None
    material_profile: str = "preview_surface"
    source_usd: Path | None = None
    material_prim_path: str | None = None
    base_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float = 0.5
    metallic: float = 0.0
    opacity: float = 1.0
    transmission: float = 0.0
    ior: float = 1.5
    thin_walled: bool = False
    textures: TextureMapSet | None = None

    def __post_init__(self) -> None:
        if self.material_id is not None:
            _non_empty_string(self.material_id, field_name="source.material_id")
        _non_empty_string(self.name, field_name="source.name")
        if self.description is not None:
            _non_empty_string(self.description, field_name="source.description")
        _non_empty_string(self.material_profile, field_name="source.material_profile")
        if (self.source_usd is None) != (self.material_prim_path is None):
            raise ValueError(
                "source.source_usd and source.material_prim_path must be provided "
                "together"
            )
        if self.source_usd is not None:
            source_usd = Path(self.source_usd)
            if not source_usd.is_absolute():
                raise ValueError("source.source_usd must be an absolute path")
            if not source_usd.is_file():
                raise FileNotFoundError(f"source USD does not exist: {source_usd}")
            _non_empty_string(
                self.material_prim_path,
                field_name="source.material_prim_path",
            )
        _base_color(self.base_color, field_name="source.base_color")
        _unit_float(self.roughness, field_name="source.roughness")
        _unit_float(self.metallic, field_name="source.metallic")
        _unit_float(self.opacity, field_name="source.opacity")
        _unit_float(self.transmission, field_name="source.transmission")
        _positive_float(self.ior, field_name="source.ior")
        _boolean(self.thin_walled, field_name="source.thin_walled")
        if self.textures is not None:
            for channel, path in (
                ("albedo", self.textures.albedo),
                ("normal", self.textures.normal),
                ("orm", self.textures.orm),
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"source texture does not exist: {channel}={path}"
                    )

    @property
    def representation(self) -> str:
        """Return the stable representation label used in evidence."""

        return "textured_pbr" if self.textures is not None else "scalar_pbr"

    @property
    def is_graph_backed(self) -> bool:
        """Return whether refinement starts from an authored USD material graph."""

        return self.source_usd is not None

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None,
        *,
        base_dir: Path,
    ) -> MaterialRefinementSource:
        values = dict(data or {})
        _reject_unknown_keys(
            values,
            allowed={
                "material_id",
                "name",
                "description",
                "material_profile",
                "source_usd",
                "material_prim_path",
                "base_color",
                "roughness",
                "metallic",
                "opacity",
                "transmission",
                "ior",
                "thin_walled",
                "textures",
            },
            field_name="source",
        )
        raw_source_usd = values.get("source_usd")
        raw_material_prim_path = values.get("material_prim_path")
        if (raw_source_usd is None) != (raw_material_prim_path is None):
            raise ValueError(
                "source.source_usd and source.material_prim_path must be provided "
                "together"
            )
        source_usd: Path | None = None
        material_prim_path: str | None = None
        graph_info = None
        if raw_source_usd is not None:
            duplicated_state = sorted(
                set(values)
                & {
                    "base_color",
                    "roughness",
                    "metallic",
                    "opacity",
                    "transmission",
                    "ior",
                    "thin_walled",
                    "textures",
                }
            )
            if duplicated_state:
                raise ValueError(
                    "source-backed material state is discovered from the authored "
                    "graph; remove: " + ", ".join(duplicated_state)
                )
            source_usd = _resolve_path(
                raw_source_usd,
                base_dir=base_dir,
                field_name="source.source_usd",
            )
            material_prim_path = _non_empty_string(
                raw_material_prim_path,
                field_name="source.material_prim_path",
            )
            from material_agent.material_library_generation.source_graph import (
                inspect_material_graph,
            )

            graph_info = inspect_material_graph(source_usd, material_prim_path)
            requested_profile = normalize_material_profile(
                values.get("material_profile", graph_info.material_profile)
            )
            if requested_profile not in {"auto", graph_info.material_profile}:
                raise ValueError(
                    "source.material_profile does not match the authored source "
                    f"graph: requested={requested_profile}, "
                    f"detected={graph_info.material_profile}"
                )
        raw_textures = values.get("textures")
        textures: TextureMapSet | None = None
        if raw_textures is not None:
            texture_values = _mapping(raw_textures, field_name="source.textures")
            _reject_unknown_keys(
                texture_values,
                allowed={"albedo", "normal", "orm"},
                field_name="source.textures",
            )
            missing = sorted(
                channel
                for channel in ("albedo", "normal", "orm")
                if channel not in texture_values
            )
            if missing:
                raise ValueError(
                    "source.textures must preserve a complete albedo/normal/orm "
                    f"set; missing: {', '.join(missing)}"
                )
            textures = TextureMapSet(
                albedo=_resolve_path(
                    texture_values["albedo"],
                    base_dir=base_dir,
                    field_name="source.textures.albedo",
                ),
                normal=_resolve_path(
                    texture_values["normal"],
                    base_dir=base_dir,
                    field_name="source.textures.normal",
                ),
                orm=_resolve_path(
                    texture_values["orm"],
                    base_dir=base_dir,
                    field_name="source.textures.orm",
                ),
            )
        if graph_info is not None:
            textures = graph_info.textures
        return cls(
            material_id=_optional_string(
                values.get("material_id"), field_name="source.material_id"
            ),
            name=_non_empty_string(
                values.get("name", "Source Material"), field_name="source.name"
            ),
            description=_optional_string(
                values.get("description"), field_name="source.description"
            ),
            material_profile=_non_empty_string(
                (
                    graph_info.material_profile
                    if graph_info is not None
                    else values.get("material_profile", "preview_surface")
                ),
                field_name="source.material_profile",
            ),
            source_usd=source_usd,
            material_prim_path=material_prim_path,
            base_color=_base_color(
                (
                    graph_info.base_color
                    if graph_info is not None
                    else values.get("base_color", (0.5, 0.5, 0.5))
                ),
                field_name="source.base_color",
            ),
            roughness=_unit_float(
                (
                    graph_info.roughness
                    if graph_info is not None
                    else values.get("roughness", 0.5)
                ),
                field_name="source.roughness",
            ),
            metallic=_unit_float(
                (
                    graph_info.metallic
                    if graph_info is not None
                    else values.get("metallic", 0.0)
                ),
                field_name="source.metallic",
            ),
            opacity=_unit_float(
                (
                    graph_info.opacity
                    if graph_info is not None
                    else values.get("opacity", 1.0)
                ),
                field_name="source.opacity",
            ),
            transmission=_unit_float(
                (
                    graph_info.transmission
                    if graph_info is not None
                    else values.get("transmission", 0.0)
                ),
                field_name="source.transmission",
            ),
            ior=_positive_float(
                graph_info.ior if graph_info is not None else values.get("ior", 1.5),
                field_name="source.ior",
            ),
            thin_walled=_boolean(
                (
                    graph_info.thin_walled
                    if graph_info is not None
                    else values.get("thin_walled", False)
                ),
                field_name="source.thin_walled",
            ),
            textures=textures,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return redaction-safe source representation evidence."""

        return {
            "representation": self.representation,
            "material_id": self.material_id,
            "name": self.name,
            "description": self.description,
            "material_profile": self.material_profile,
            "source_usd": (
                self.source_usd.as_posix() if self.source_usd is not None else None
            ),
            "material_prim_path": self.material_prim_path,
            "base_color": list(self.base_color),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "opacity": self.opacity,
            "transmission": self.transmission,
            "ior": self.ior,
            "thin_walled": self.thin_walled,
            "textures": (
                {
                    "albedo": self.textures.albedo.as_posix(),
                    "normal": self.textures.normal.as_posix(),
                    "orm": self.textures.orm.as_posix(),
                }
                if self.textures is not None
                else None
            ),
        }


@dataclass(frozen=True)
class MaterialRefinementConfig:
    """Resolved input for one local material-refinement run."""

    goal: MaterialRefinementGoal
    output_dir: Path
    source: MaterialRefinementSource
    variation: TextureVariationSettings
    search: MaterialRefinementSearchSettings
    optimizer: OptimizerSettings
    render: MaterialRefinementRenderSettings
    judge: MaterialRefinementJudgeSettings
    max_refinements: int = 2
    overwrite: bool = False

    def __post_init__(self) -> None:
        _non_negative_int(
            self.max_refinements, field_name="optimization.max_refinements"
        )

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path | None = None,
        output_dir_override: Path | None = None,
    ) -> MaterialRefinementConfig:
        root = _mapping(data, field_name="material refinement config")
        _reject_unknown_keys(
            root,
            allowed={
                "goal",
                "source",
                "variation",
                "search",
                "optimization",
                "render",
                "judge",
                "output_dir",
                "overwrite",
                "variation_set",
            },
            field_name="material refinement config",
        )
        resolved_base = (base_dir or Path.cwd()).expanduser().resolve()
        goal = MaterialRefinementGoal.from_mapping(
            _mapping(root.get("goal"), field_name="goal"),
            base_dir=resolved_base,
        )
        source = MaterialRefinementSource.from_mapping(
            _mapping(root.get("source"), field_name="source"),
            base_dir=resolved_base,
        )

        optimization_data = _mapping(
            root.get("optimization", {}), field_name="optimization"
        )
        _reject_unknown_keys(
            optimization_data,
            allowed={
                "name",
                "max_trials",
                "seed",
                "replicas",
                "replica_seed",
                "max_refinements",
            },
            field_name="optimization",
        )
        raw_replica_seed = optimization_data.get("replica_seed")
        optimizer = OptimizerSettings(
            name=_non_empty_string(
                optimization_data.get("name", OPTIMIZER_AUTO),
                field_name="optimization.name",
            ),
            max_trials=_positive_int(
                optimization_data.get("max_trials", 12),
                field_name="optimization.max_trials",
            ),
            seed=_non_negative_int(
                optimization_data.get("seed", 42), field_name="optimization.seed"
            ),
            replicas=_positive_int(
                optimization_data.get("replicas", 1),
                field_name="optimization.replicas",
            ),
            replica_seed=(
                _non_negative_int(
                    raw_replica_seed, field_name="optimization.replica_seed"
                )
                if raw_replica_seed is not None
                else None
            ),
        )
        output_dir = _resolve_path(
            (
                root.get("output_dir", _DEFAULT_OUTPUT_DIR)
                if output_dir_override is None
                else output_dir_override
            ),
            base_dir=resolved_base,
            field_name="output_dir",
        )
        return cls(
            goal=goal,
            output_dir=output_dir,
            source=source,
            variation=TextureVariationSettings.from_mapping(
                _mapping(root.get("variation", {}), field_name="variation")
            ),
            search=MaterialRefinementSearchSettings.from_mapping(
                _mapping(root.get("search", {}), field_name="search")
            ),
            optimizer=optimizer,
            render=MaterialRefinementRenderSettings.from_mapping(
                _mapping(root.get("render", {}), field_name="render")
            ),
            judge=MaterialRefinementJudgeSettings.from_mapping(
                _mapping(root.get("judge", {}), field_name="judge")
            ),
            max_refinements=_non_negative_int(
                optimization_data.get("max_refinements", 2),
                field_name="optimization.max_refinements",
            ),
            overwrite=_boolean(root.get("overwrite", False), field_name="overwrite"),
        )

    @property
    def material_profile(self) -> str:
        """Return the source profile that candidate outputs must preserve."""

        return self.source.material_profile

    @property
    def search_params(self) -> tuple[TunableParam, ...]:
        """Return source-anchored material controls plus service strength."""

        return self.search.parameters_for_values(
            base_color=self.source.base_color,
            roughness=self.source.roughness,
            metallic=self.source.metallic,
            variation=self.variation,
            include_texture_variation=self.source.textures is not None,
        )


__all__ = [
    "MaterialObjectiveWeights",
    "MaterialRefinementConfig",
    "MaterialRefinementGoal",
    "MaterialRefinementJudgeSettings",
    "MaterialRefinementRenderSettings",
    "MaterialRefinementSearchSettings",
    "MaterialRefinementSource",
    "TextureVariationSettings",
]
