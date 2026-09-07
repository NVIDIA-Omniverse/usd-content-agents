# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Variation-set orchestration built exclusively from one-material refinement."""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from material_agent.material_library_generation.manifests import (
    write_generation_plan,
    write_materials_manifest,
)
from material_agent.material_library_generation.schema import (
    GeneratedMaterial,
    MaterialGenerationPlan,
    MaterialRecipeSemantics,
    TextureMapSet,
    make_material_id,
)
from material_agent.material_library_generation.source_graph import (
    MaterialGraphEdit,
    write_edited_material_graphs,
)
from material_agent.material_library_generation.usd_authoring import (
    write_material_library_usd,
)

from .artifacts import artifact_sha256
from .authoring_adapter import candidate_material_recipe, source_material_recipe
from .contracts import MaterialRefinementConfig
from .objective import MaterialFeatures, extract_material_features
from .runner import (
    MaterialRefinementCancelled,
    MaterialRefinementError,
    MaterialRefinementResult,
    run_material_refinement,
)
from .texture_variation import TextureVariationGenerator
from .visual import RenderTaskLike, VlmJudgeLike

_OUTPUT_MARKER = ".material-variation-output"
_SCHEMA_VERSION = "material-agent-variation.v1"
_MAX_VARIANTS = 16

VariationTrialCallback = Callable[[int, int, Any], None]


def _selected_material_to_dict(material: GeneratedMaterial) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": material.name,
        "binding": material.binding,
        "representation": "textured_pbr" if material.is_textured else "scalar_pbr",
        "pbr": {
            "base_color": list(material.recipe.base_color_hint),
            **material.recipe.pbr_hints.to_dict(),
        },
    }
    if material.textures is not None:
        entry["textures"] = {
            "albedo": {
                "path": material.textures.albedo.as_posix(),
                "sha256": artifact_sha256(material.textures.albedo),
            },
            "normal": {
                "path": material.textures.normal.as_posix(),
                "sha256": artifact_sha256(material.textures.normal),
            },
            "orm": {
                "path": material.textures.orm.as_posix(),
                "sha256": artifact_sha256(material.textures.orm),
            },
        }
    return entry


def _reject_unknown_variation_keys(values: dict[str, Any]) -> None:
    allowed = {
        "description",
        "requested_count",
        "minimum_feature_distance",
        "diversity_weight",
    }
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        raise ValueError(
            "variation_set contains unknown field(s): " + ", ".join(unknown)
        )


@dataclass(frozen=True)
class MaterialVariationGoal:
    """Diversity requirements layered over one directed refinement goal."""

    description: str = (
        "Create a visibly distinct surface treatment while preserving material "
        "identity and physical plausibility."
    )
    requested_count: int = 3
    minimum_feature_distance: float = 0.05
    diversity_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("variation_set.description must not be empty")
        if (
            isinstance(self.requested_count, bool)
            or not isinstance(self.requested_count, int)
            or not 1 <= self.requested_count <= _MAX_VARIANTS
        ):
            raise ValueError(
                f"variation_set.requested_count must be in [1, {_MAX_VARIANTS}]"
            )
        if isinstance(self.minimum_feature_distance, bool) or not isinstance(
            self.minimum_feature_distance, int | float
        ):
            raise TypeError("variation_set.minimum_feature_distance must be a number")
        if not math.isfinite(self.minimum_feature_distance) or not (
            0.0 <= self.minimum_feature_distance <= 1.0
        ):
            raise ValueError(
                "variation_set.minimum_feature_distance must be finite and in [0, 1]"
            )
        if isinstance(self.diversity_weight, bool) or not isinstance(
            self.diversity_weight, int | float
        ):
            raise TypeError("variation_set.diversity_weight must be a number")
        if not math.isfinite(self.diversity_weight) or self.diversity_weight < 0.0:
            raise ValueError(
                "variation_set.diversity_weight must be finite and non-negative"
            )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> MaterialVariationGoal:
        values = dict(data or {})
        _reject_unknown_variation_keys(values)
        return cls(
            description=values.get("description", cls.description),
            requested_count=values.get("requested_count", 3),
            minimum_feature_distance=values.get("minimum_feature_distance", 0.05),
            diversity_weight=values.get("diversity_weight", 1.0),
        )


@dataclass(frozen=True)
class MaterialVariationConfig:
    """Resolved variation-set request and reusable refinement configuration."""

    refinement: MaterialRefinementConfig
    variation_goal: MaterialVariationGoal

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        base_dir: Path | None = None,
        output_dir_override: Path | None = None,
    ) -> MaterialVariationConfig:
        refinement = MaterialRefinementConfig.from_mapping(
            data,
            base_dir=base_dir,
            output_dir_override=output_dir_override,
        )
        raw_variation = data.get("variation_set", {})
        if not isinstance(raw_variation, dict):
            raise TypeError("variation_set must be a mapping")
        return cls(
            refinement=refinement,
            variation_goal=MaterialVariationGoal.from_mapping(raw_variation),
        )


@dataclass(frozen=True)
class MaterialVariationSlot:
    """One slot's invocation of the shared one-material refinement core."""

    slot: int
    status: Literal["completed", "incomplete", "cancelled"]
    result: MaterialRefinementResult | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "status": self.status,
            "error": self.error,
            "result": self.result.to_dict() if self.result is not None else None,
        }


@dataclass(frozen=True)
class MaterialVariationResult:
    """Published variation library or auditable incomplete slot history."""

    status: Literal["completed", "incomplete", "cancelled"]
    termination_reason: str
    slots: tuple[MaterialVariationSlot, ...]
    selected_materials: tuple[GeneratedMaterial, ...]
    selected_render_paths: tuple[Path, ...]
    manifest_path: Path
    material_library_path: Path | None = None
    materials_manifest_path: Path | None = None
    generation_plan_path: Path | None = None

    @property
    def success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "status": self.status,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "slots": [slot.to_dict() for slot in self.slots],
            "selected_materials": [
                _selected_material_to_dict(material)
                for material in self.selected_materials
            ],
            "selected_render_paths": [
                path.as_posix() for path in self.selected_render_paths
            ],
            "manifest_path": self.manifest_path.as_posix(),
            "material_library_path": (
                self.material_library_path.as_posix()
                if self.material_library_path is not None
                else None
            ),
            "materials_manifest_path": (
                self.materials_manifest_path.as_posix()
                if self.materials_manifest_path is not None
                else None
            ),
            "generation_plan_path": (
                self.generation_plan_path.as_posix()
                if self.generation_plan_path is not None
                else None
            ),
            "published_artifact_sha256": {
                name: artifact_sha256(path)
                for name, path in (
                    ("material_library", self.material_library_path),
                    ("materials_manifest", self.materials_manifest_path),
                    ("generation_plan", self.generation_plan_path),
                )
                if path is not None and path.is_file()
            },
        }


def _prepare_output(config: MaterialVariationConfig) -> Path:
    output_dir = config.refinement.output_dir
    if output_dir == Path(output_dir.anchor):
        raise ValueError("material variation output cannot be a filesystem root")
    marker = output_dir / _OUTPUT_MARKER
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError("material variation output is not a directory")
        if any(output_dir.iterdir()):
            if config.refinement.overwrite and marker.is_file():
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    "material variation output already exists and is not an "
                    "overwrite-safe variation directory"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{_SCHEMA_VERSION}\n", encoding="ascii")
    return output_dir


def _slot_prompt(
    config: MaterialVariationConfig,
    *,
    slot: int,
    prior_count: int,
) -> str:
    prior = (
        f"This set already has {prior_count} approved variation(s). Produce a "
        "clearly non-duplicate treatment."
        if prior_count
        else "This is the first variation in the set."
    )
    return (
        f"{config.refinement.goal.appearance_prompt}\n\n"
        f"Variation-set goal: {config.variation_goal.description}\n"
        f"Requested slot: {slot} of {config.variation_goal.requested_count}.\n"
        f"{prior}"
    )


def _slot_config(
    config: MaterialVariationConfig, slot: int
) -> MaterialRefinementConfig:
    refinement = config.refinement
    sweeps_per_slot = refinement.max_refinements + 1
    seed_offset = (slot - 1) * sweeps_per_slot * refinement.optimizer.max_trials
    optimizer = replace(
        refinement.optimizer,
        seed=refinement.optimizer.seed + seed_offset,
        replica_seed=(
            refinement.optimizer.replica_seed + seed_offset
            if refinement.optimizer.replica_seed is not None
            else None
        ),
    )
    return replace(
        refinement,
        goal=replace(
            refinement.goal,
            appearance_prompt=_slot_prompt(config, slot=slot, prior_count=slot - 1),
        ),
        optimizer=optimizer,
        output_dir=refinement.output_dir / "slots" / f"slot_{slot:03d}",
        overwrite=False,
    )


def _copy_selected_material(
    result: MaterialRefinementResult,
    config: MaterialVariationConfig,
    *,
    slot: int,
    final_dir: Path,
) -> tuple[GeneratedMaterial, MaterialFeatures]:
    source = config.refinement.source
    source_recipe = source_material_recipe(source)
    material_id = make_material_id(f"{source_recipe.material_id}_variation_{slot:03d}")
    name = f"{source_recipe.name} Variation {slot}"
    textures: TextureMapSet | None = None
    if config.refinement.source.textures is not None:
        texture_dir = final_dir / "textures" / material_id
        texture_dir.mkdir(parents=True, exist_ok=True)
        textures = TextureMapSet(
            albedo=texture_dir / "albedo.png",
            normal=texture_dir / "normal.png",
            orm=texture_dir / "orm.png",
        )
        for channel, destination in (
            ("albedo", textures.albedo),
            ("normal", textures.normal),
            ("orm", textures.orm),
        ):
            shutil.copy2(result.best_artifacts[channel], destination)
        features = extract_material_features(
            albedo_path=textures.albedo, orm_path=textures.orm
        )
    else:
        features = MaterialFeatures(
            base_color=(
                result.best_params["base_color_r"],
                result.best_params["base_color_g"],
                result.best_params["base_color_b"],
            ),
            roughness=result.best_params["roughness"],
            metallic=result.best_params["metallic"],
            color_source="authored_pbr",
        )
    recipe = candidate_material_recipe(
        source,
        material_id=material_id,
        name=name,
        description=f"{source_recipe.description} ({config.variation_goal.description})",
        appearance_prompt=_slot_prompt(config, slot=slot, prior_count=slot - 1),
        base_color=features.base_color or source.base_color,
        roughness=(
            features.roughness if features.roughness is not None else source.roughness
        ),
        metallic=(
            features.metallic if features.metallic is not None else source.metallic
        ),
    )
    prototype_source = None
    if source.is_graph_backed:
        prototype_source = {
            "library_path": result.best_artifacts["material_usd"].as_posix(),
            "binding": source_recipe.binding,
            "material_profile": source.material_profile,
        }
    return (
        GeneratedMaterial(
            recipe=recipe,
            textures=textures,
            prototype_source=prototype_source,
        ),
        features,
    )


def _write_manifest(path: Path, result: MaterialVariationResult) -> None:
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_material_variations(
    config: MaterialVariationConfig,
    *,
    cancel_check: Callable[[], bool] | None = None,
    on_trial: VariationTrialCallback | None = None,
    generator: TextureVariationGenerator | None = None,
    render_task: RenderTaskLike | None = None,
    rendering_backend: Any | None = None,
    vlm_judge: VlmJudgeLike | None = None,
) -> MaterialVariationResult:
    """Refine one material per slot and carry approved diversity evidence forward."""

    output_dir = _prepare_output(config)
    manifest_path = output_dir / "material_variation_manifest.json"
    slots: list[MaterialVariationSlot] = []
    selected_results: list[MaterialRefinementResult] = []
    selected_features: list[MaterialFeatures] = []
    selected_renders: list[Path] = []

    for slot in range(1, config.variation_goal.requested_count + 1):
        if cancel_check is not None and cancel_check():
            slots.append(
                MaterialVariationSlot(
                    slot=slot,
                    status="cancelled",
                    result=None,
                    error="material variation cancelled by caller",
                )
            )
            break
        slot_config = _slot_config(config, slot)

        def slot_trial(attempt: int, trial: Any, _slot: int = slot) -> None:
            if on_trial is not None:
                on_trial(_slot, attempt, trial)

        try:
            result = run_material_refinement(
                slot_config,
                cancel_check=cancel_check,
                on_trial=slot_trial,
                generator=generator,
                render_task=render_task,
                rendering_backend=rendering_backend,
                vlm_judge=vlm_judge,
                diversity_features=tuple(selected_features),
                comparison_image_paths=tuple(selected_renders),
                minimum_diversity=config.variation_goal.minimum_feature_distance,
                diversity_weight=config.variation_goal.diversity_weight,
            )
        except MaterialRefinementCancelled as error:
            slots.append(
                MaterialVariationSlot(
                    slot=slot, status="cancelled", result=None, error=str(error)
                )
            )
            break
        except MaterialRefinementError as error:
            slots.append(
                MaterialVariationSlot(
                    slot=slot, status="incomplete", result=None, error=str(error)
                )
            )
            break
        if result.termination_reason == "cancelled":
            slots.append(
                MaterialVariationSlot(
                    slot=slot,
                    status="cancelled",
                    result=result,
                    error="material refinement was cancelled after visual judgment",
                )
            )
            break
        if not result.approved:
            slots.append(
                MaterialVariationSlot(
                    slot=slot,
                    status="incomplete",
                    result=result,
                    error="material refinement reached its limit without approval",
                )
            )
            break
        slots.append(
            MaterialVariationSlot(slot=slot, status="completed", result=result)
        )
        selected_results.append(result)
        selected_features.append(
            extract_material_features(
                albedo_path=result.best_artifacts["albedo"],
                orm_path=result.best_artifacts["orm"],
            )
            if config.refinement.source.textures is not None
            else MaterialFeatures(
                base_color=(
                    result.best_params["base_color_r"],
                    result.best_params["base_color_g"],
                    result.best_params["base_color_b"],
                ),
                roughness=result.best_params["roughness"],
                metallic=result.best_params["metallic"],
                color_source="authored_pbr",
            )
        )
        selected_renders.append(result.best_artifacts["rendered_image_1"])

    complete = len(selected_results) == config.variation_goal.requested_count
    cancelled = bool(slots and slots[-1].status == "cancelled")
    if not complete:
        variation_result = MaterialVariationResult(
            status="cancelled" if cancelled else "incomplete",
            termination_reason="cancelled" if cancelled else "slot_incomplete",
            slots=tuple(slots),
            selected_materials=(),
            selected_render_paths=tuple(selected_renders),
            manifest_path=manifest_path,
        )
        _write_manifest(manifest_path, variation_result)
        return variation_result

    final_dir = output_dir / "final"
    selected_materials: list[GeneratedMaterial] = []
    for slot, selected in enumerate(selected_results, start=1):
        material, _features = _copy_selected_material(
            selected, config, slot=slot, final_dir=final_dir
        )
        selected_materials.append(material)
    material_library_path = final_dir / "material_library.usda"
    materials_manifest_path = final_dir / "materials.yaml"
    generation_plan_path = final_dir / "material_variation_plan.yaml"
    if config.refinement.source.is_graph_backed:
        graph_edits: list[MaterialGraphEdit] = []
        for material in selected_materials:
            prototype = material.prototype_source or {}
            source_library = prototype.get("library_path")
            source_binding = prototype.get("binding")
            if not source_library or not source_binding:
                raise ValueError(
                    "source-backed variation is missing its selected material graph"
                )
            graph_edits.append(
                MaterialGraphEdit(
                    source_usd=Path(str(source_library)),
                    source_material_prim_path=str(source_binding),
                    target_material_prim_path=material.binding,
                    base_color=material.recipe.base_color_hint,
                    roughness=material.recipe.pbr_hints.roughness,
                    metallic=material.recipe.pbr_hints.metallic,
                    textures=material.textures,
                )
            )
        write_edited_material_graphs(material_library_path, graph_edits)
    else:
        write_material_library_usd(
            material_library_path,
            tuple(selected_materials),
            material_profile=config.refinement.material_profile,
            recipe_semantics=MaterialRecipeSemantics.LITERAL_SHADER_VALUES,
        )
    write_materials_manifest(
        materials_manifest_path, material_library_path, tuple(selected_materials)
    )
    write_generation_plan(
        generation_plan_path,
        MaterialGenerationPlan(
            materials=tuple(material.recipe for material in selected_materials),
            asset={
                "workflow": "material_variation",
                "goal": config.variation_goal.description,
            },
        ),
        tuple(selected_materials),
    )
    variation_result = MaterialVariationResult(
        status="completed",
        termination_reason="requested_count_reached",
        slots=tuple(slots),
        selected_materials=tuple(selected_materials),
        selected_render_paths=tuple(selected_renders),
        manifest_path=manifest_path,
        material_library_path=material_library_path,
        materials_manifest_path=materials_manifest_path,
        generation_plan_path=generation_plan_path,
    )
    _write_manifest(manifest_path, variation_result)
    return variation_result


__all__ = [
    "MaterialVariationConfig",
    "MaterialVariationGoal",
    "MaterialVariationResult",
    "MaterialVariationSlot",
    "run_material_variations",
]
