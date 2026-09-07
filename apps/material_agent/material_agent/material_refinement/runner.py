# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material adapter for shared bounded tuning and iterative refinement."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from world_understanding.optimization import (
    OptimizationCancelledError,
    OptimizerUnavailableError,
    RefinementDecision,
    RefinementIteration,
    ReplicaRecord,
    TrialRecord,
    TunableParam,
    TuneRun,
    TuningObjective,
    run_refinement,
    run_tuning,
)
from world_understanding.optimization.optimizers import get_runner, resolve_optimizer
from world_understanding.utils.credentials import redact_sensitive_config

from material_agent.material_library_generation.authoring import (
    MaterialAuthoringOperation,
    MaterialAuthoringRequest,
    SourceMaterialReference,
    author_material_package,
)
from material_agent.material_library_generation.manifests import (
    write_generation_plan,
    write_materials_manifest,
)
from material_agent.material_library_generation.schema import (
    GeneratedMaterial,
    MaterialGenerationPlan,
    MaterialRecipeSemantics,
    TextureMapSet,
)
from material_agent.material_library_generation.source_graph import (
    MaterialGraphEdit,
    write_edited_material_graphs,
)
from material_agent.material_library_generation.usd_authoring import (
    write_material_library_usd,
)

from .artifacts import (
    artifact_sha256,
    materialize_candidate_maps,
    validate_material_maps,
)
from .authoring_adapter import candidate_material_recipe, source_material_recipe
from .contracts import MaterialObjectiveWeights, MaterialRefinementConfig
from .objective import (
    MaterialFeatures,
    extract_material_features,
    material_feature_distance,
    score_material_features,
)
from .texture_variation import (
    RestTextureVariationGenerator,
    TextureVariationError,
    TextureVariationGenerator,
    TextureVariationRequest,
)
from .visual import (
    MaterialJudgeVerdict,
    MaterialRenderEvidence,
    MaterialTargetInference,
    RenderTaskLike,
    VlmJudgeLike,
    author_material_swatch,
    infer_material_target,
    judge_rendered_material,
    provision_rendering_backend,
    provision_vlm_judge,
    render_material_swatch,
)

_OBJECTIVE = TuningObjective(name="material_goal_distance", unit="normalized")
_PROXY_WEIGHTS = MaterialObjectiveWeights()
_OUTPUT_MARKER = ".material-refinement-output"
_SCHEMA_VERSION = "material-agent-refinement.v1"
_PBR_CONTROL_NAMES = (
    "base_color_r",
    "base_color_g",
    "base_color_b",
    "roughness",
    "metallic",
)

CancelCheck = Callable[[], bool]
TrialCallback = Callable[[int, TrialRecord], None]


class MaterialRefinementError(RuntimeError):
    """Raised when material refinement cannot publish a judged candidate."""


class MaterialRefinementDependencyError(MaterialRefinementError):
    """Raised when an optimizer, generator, renderer, or judge is unavailable."""


class MaterialRefinementCancelled(MaterialRefinementError):
    """Raised when cancellation occurs before any candidate is judged."""


@dataclass(frozen=True)
class MaterialSearchSpace:
    """Bounded Texture Variation API controls tuned by the shared optimizer."""

    params: tuple[TunableParam, ...]

    def to_dict(self) -> dict[str, dict[str, float | bool]]:
        return {
            parameter.name: {
                "min": parameter.min_value,
                "max": parameter.max_value,
                "integer": parameter.integer,
            }
            for parameter in self.params
        }

    def bounds(self, name: str) -> tuple[float, float]:
        """Return one named parameter's inclusive bounds."""

        parameter = next(item for item in self.params if item.name == name)
        return parameter.min_value, parameter.max_value


def _merge_search_spaces(
    current: MaterialSearchSpace,
    proposed: MaterialSearchSpace,
) -> MaterialSearchSpace:
    """Expand current bounds to include one target-centered proposal."""

    proposed_by_name = {parameter.name: parameter for parameter in proposed.params}
    if {parameter.name for parameter in current.params} != set(proposed_by_name):
        raise MaterialRefinementError("material search-space controls changed")
    merged: list[TunableParam] = []
    for parameter in current.params:
        proposal = proposed_by_name[parameter.name]
        if parameter.integer != proposal.integer:
            raise MaterialRefinementError(
                f"material search-space type changed for {parameter.name!r}"
            )
        merged.append(
            TunableParam(
                name=parameter.name,
                min_value=min(parameter.min_value, proposal.min_value),
                max_value=max(parameter.max_value, proposal.max_value),
                integer=parameter.integer,
            )
        )
    return MaterialSearchSpace(tuple(merged))


def _changed_search_bounds(
    current: MaterialSearchSpace,
    revised: MaterialSearchSpace,
) -> tuple[str, ...]:
    return tuple(
        parameter.name
        for parameter in current.params
        if current.bounds(parameter.name) != revised.bounds(parameter.name)
    )


def _semantic_pbr_control_bounds() -> dict[str, tuple[float, float]]:
    """Return normalized bounds for prompt-derived material targets."""

    return dict.fromkeys(_PBR_CONTROL_NAMES, (0.0, 1.0))


@dataclass(frozen=True)
class _RefinementState:
    prompt: str
    incumbent: dict[str, float]
    target_features: MaterialFeatures
    target_inference: MaterialTargetInference
    search_space: MaterialSearchSpace
    previous_feedback: str | None = None


@dataclass(frozen=True)
class _SweepEvaluation:
    iteration: int
    state: _RefinementState
    tuning: TuneRun
    best_trial: TrialRecord | None
    representative: ReplicaRecord | None
    iteration_dir: Path


@dataclass(frozen=True)
class _SweepJudgment:
    render: MaterialRenderEvidence | None = None
    verdict: MaterialJudgeVerdict | None = None
    target_revision: MaterialTargetInference | None = None
    revised_search_space: MaterialSearchSpace | None = None
    evidence_dir: Path | None = None
    error: str | None = None
    target_revision_error: str | None = None
    cancelled: bool = False


@dataclass(frozen=True)
class _JudgedCandidate:
    iteration: int
    trial: TrialRecord
    representative: ReplicaRecord
    render: MaterialRenderEvidence
    verdict: MaterialJudgeVerdict
    target_features: MaterialFeatures
    search_space: MaterialSearchSpace
    evidence_dir: Path


@dataclass(frozen=True)
class MaterialRefinementAttempt:
    """Summary of one fresh optimizer sweep and winner-level visual review."""

    iteration: int
    optimizer_seed: int
    prompt: str
    target_features: MaterialFeatures
    search_space: MaterialSearchSpace
    trial_count: int
    best_params: dict[str, float]
    best_score: float | None
    representative_score: float | None
    representative_seed: int | None
    approved: bool
    cancelled: bool
    render_evidence: MaterialRenderEvidence | None
    judge_verdict: MaterialJudgeVerdict | None
    target_revision: MaterialTargetInference | None
    revised_search_space: MaterialSearchSpace | None
    error: str | None
    result_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "optimizer_seed": self.optimizer_seed,
            "prompt": self.prompt,
            "target_features": self.target_features.to_dict(),
            "search_space": self.search_space.to_dict(),
            "trial_count": self.trial_count,
            "best_params": dict(self.best_params),
            "best_score": self.best_score,
            "representative_score": self.representative_score,
            "representative_seed": self.representative_seed,
            "approved": self.approved,
            "cancelled": self.cancelled,
            "render_evidence": (
                self.render_evidence.to_dict()
                if self.render_evidence is not None
                else None
            ),
            "judge_verdict": (
                self.judge_verdict.to_dict() if self.judge_verdict is not None else None
            ),
            "target_revision": (
                self.target_revision.to_dict()
                if self.target_revision is not None
                else None
            ),
            "revised_search_space": (
                self.revised_search_space.to_dict()
                if self.revised_search_space is not None
                else None
            ),
            "error": self.error,
            "result_path": self.result_path.as_posix(),
        }


@dataclass(frozen=True)
class MaterialRefinementResult:
    """Published material and evidence from a terminal refinement run."""

    termination_reason: str
    optimizer: str
    best_params: dict[str, float]
    best_score: float
    representative_score: float
    representative_seed: int
    judge_score: float
    judge_decision: str
    best_material_dir: Path
    best_evidence_dir: Path
    best_artifacts: dict[str, Path]
    target_features: MaterialFeatures
    search_space: MaterialSearchSpace
    source_material_dir: Path
    source_swatch_path: Path
    history_path: Path
    summary_path: Path
    attempts: tuple[MaterialRefinementAttempt, ...]
    trial_count: int

    @property
    def approved(self) -> bool:
        return self.termination_reason == "approved"

    @property
    def success(self) -> bool:
        return self.termination_reason in {"approved", "max_iterations"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "termination_reason": self.termination_reason,
            "approved": self.approved,
            "optimizer": self.optimizer,
            "best_params": dict(self.best_params),
            "best_score": self.best_score,
            "representative_score": self.representative_score,
            "representative_seed": self.representative_seed,
            "judge_score": self.judge_score,
            "judge_decision": self.judge_decision,
            "best_material_dir": self.best_material_dir.as_posix(),
            "best_evidence_dir": self.best_evidence_dir.as_posix(),
            "best_artifacts": {
                name: path.as_posix() for name, path in self.best_artifacts.items()
            },
            "best_artifact_sha256": {
                name: artifact_sha256(path)
                for name, path in self.best_artifacts.items()
                if path.is_file()
            },
            "target_features": self.target_features.to_dict(),
            "search_space": self.search_space.to_dict(),
            "source_material_dir": self.source_material_dir.as_posix(),
            "source_swatch_path": self.source_swatch_path.as_posix(),
            "history_path": self.history_path.as_posix(),
            "history_sha256": artifact_sha256(self.history_path),
            "summary_path": self.summary_path.as_posix(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "trial_count": self.trial_count,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _safe_error(error: Exception) -> str:
    projected = redact_sensitive_config(str(error))
    if isinstance(projected, str) and projected.strip():
        return projected
    return type(error).__name__


def _check_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise OptimizationCancelledError("material refinement cancelled by caller")


def _prepare_output_dir(config: MaterialRefinementConfig) -> None:
    output_dir = config.output_dir
    if output_dir == Path(output_dir.anchor):
        raise ValueError("material refinement output cannot be a filesystem root")
    marker = output_dir / _OUTPUT_MARKER
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError("material refinement output is not a directory")
        if any(output_dir.iterdir()):
            if config.overwrite and marker.is_file():
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(
                    "material refinement output already exists and is not an "
                    "overwrite-safe refinement directory"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{_SCHEMA_VERSION}\n", encoding="ascii")


def _build_source_material(
    config: MaterialRefinementConfig,
) -> tuple[Path, Path, MaterialFeatures]:
    source_dir = config.output_dir / "source"
    package_dir = source_dir / "material"
    package_dir.mkdir(parents=True, exist_ok=True)
    source_recipe = source_material_recipe(config.source)
    textures: TextureMapSet | None = None
    source_hash_before: str | None = None
    prototype_source: dict[str, Any] | None = None
    if config.source.is_graph_backed:
        if config.source.source_usd is None or config.source.material_prim_path is None:
            raise MaterialRefinementError("source graph contract is incomplete")
        source_hash_before = artifact_sha256(config.source.source_usd)
        (published_graph,) = write_edited_material_graphs(
            package_dir / "material_library.usda",
            (
                MaterialGraphEdit(
                    source_usd=config.source.source_usd,
                    source_material_prim_path=config.source.material_prim_path,
                    target_material_prim_path=source_recipe.binding,
                ),
            ),
        )
        textures = published_graph.textures
        prototype_source = {
            "library_path": config.source.source_usd.as_posix(),
            "binding": config.source.material_prim_path,
            "material_profile": published_graph.material_profile,
            "topology_sha256": published_graph.topology_sha256,
        }
        source_features = MaterialFeatures(
            base_color=published_graph.base_color,
            roughness=published_graph.roughness,
            metallic=published_graph.metallic,
            color_source=(
                "source_graph_textures"
                if published_graph.textures is not None
                else "source_graph_inputs"
            ),
        )
        library_path = package_dir / "material_library.usda"
    elif config.source.textures is not None:
        texture_dir = package_dir / "textures" / source_recipe.material_id
        texture_dir.mkdir(parents=True, exist_ok=True)

        def copy_texture(channel: str, source: Path) -> Path:
            suffix = source.suffix.lower() or ".png"
            destination: Path = texture_dir / f"{channel}{suffix}"
            shutil.copy2(source, destination)
            return destination

        textures = TextureMapSet(
            albedo=copy_texture("albedo", config.source.textures.albedo),
            normal=copy_texture("normal", config.source.textures.normal),
            orm=copy_texture("orm", config.source.textures.orm),
        )
        source_material = GeneratedMaterial(recipe=source_recipe, textures=textures)
        library_path = write_material_library_usd(
            package_dir / "material_library.usda",
            (source_material,),
            material_profile=config.material_profile,
            recipe_semantics=MaterialRecipeSemantics.LITERAL_SHADER_VALUES,
        )
        source_features = extract_material_features(
            albedo_path=textures.albedo,
            orm_path=textures.orm,
        )
    else:
        source_material = GeneratedMaterial(recipe=source_recipe, textures=None)
        library_path = write_material_library_usd(
            package_dir / "material_library.usda",
            (source_material,),
            material_profile=config.material_profile,
            recipe_semantics=MaterialRecipeSemantics.LITERAL_SHADER_VALUES,
        )
        source_features = MaterialFeatures(
            base_color=config.source.base_color,
            roughness=config.source.roughness,
            metallic=config.source.metallic,
            color_source="source_pbr",
        )
    plan = MaterialGenerationPlan(
        materials=(source_recipe,),
        asset={
            "workflow": "material_refinement",
            "role": "source",
            "source_backed": config.source.is_graph_backed,
        },
    )
    source_material = GeneratedMaterial(
        recipe=source_recipe,
        textures=textures,
        prototype_source=prototype_source,
    )
    materials_manifest_path = write_materials_manifest(
        package_dir / "materials.yaml", library_path, (source_material,)
    )
    generation_plan_path = write_generation_plan(
        package_dir / "material_generation_plan.yaml",
        plan,
        (source_material,),
    )
    map_validation = validate_material_maps(textures) if textures is not None else None
    if source_hash_before is not None and config.source.source_usd is not None:
        if artifact_sha256(config.source.source_usd) != source_hash_before:
            raise MaterialRefinementError(
                "source USD changed while preparing refinement evidence"
            )
    swatch_path = author_material_swatch(
        material_usd_path=library_path,
        material_binding=source_recipe.binding,
        output_path=source_dir / "source_swatch.usda",
    )
    _write_json(
        source_dir / "source.json",
        {
            "source": config.source.to_dict(),
            "authoring_material": {
                "material_id": source_recipe.material_id,
                "binding": source_recipe.binding,
            },
            "material_library": library_path.as_posix(),
            "materials_manifest": materials_manifest_path.as_posix(),
            "generation_plan": generation_plan_path.as_posix(),
            "swatch_usd": swatch_path.as_posix(),
            "textures": (
                {
                    "albedo": textures.albedo.as_posix(),
                    "normal": textures.normal.as_posix(),
                    "orm": textures.orm.as_posix(),
                }
                if textures is not None
                else None
            ),
            "map_validation": (
                map_validation.to_dict() if map_validation is not None else None
            ),
            "source_graph": prototype_source,
            "features": source_features.to_dict(),
            "artifact_sha256": {
                "material_library": artifact_sha256(library_path),
                "materials_manifest": artifact_sha256(materials_manifest_path),
                "generation_plan": artifact_sha256(generation_plan_path),
                "swatch_usd": artifact_sha256(swatch_path),
            },
        },
    )
    return package_dir, swatch_path, source_features


def _iteration_settings(config: MaterialRefinementConfig, iteration: int) -> Any:
    offset = (iteration - 1) * config.optimizer.max_trials
    replica_seed = config.optimizer.replica_seed
    if replica_seed is not None:
        replica_seed += offset
    return replace(
        config.optimizer,
        seed=config.optimizer.seed + offset,
        replica_seed=replica_seed,
    )


def _candidate_prompt(
    base_prompt: str,
    *,
    iteration: int,
    feedback: str | None,
) -> str:
    if not feedback:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        f"Visual review from refinement attempt {iteration - 1}:\n{feedback}\n\n"
        "Create another variation that directly addresses that feedback while "
        "preserving the source material's intended identity and physical use."
    )


def _goal_generation_prompt(config: MaterialRefinementConfig) -> str:
    return config.goal.appearance_prompt


def _trial_generation_prompt(prompt: str, params: dict[str, float]) -> str:
    material_controls = {
        "base_color": [
            params["base_color_r"],
            params["base_color_g"],
            params["base_color_b"],
        ],
        "roughness": params["roughness"],
        "metallic": params["metallic"],
    }
    return (
        f"{prompt}\n\n"
        "This optimizer trial must target these normalized PBR controls: "
        f"{json.dumps(material_controls, sort_keys=True)}"
    )


def _publish_candidate_package(
    *,
    config: MaterialRefinementConfig,
    package_dir: Path,
    textures: TextureMapSet | None,
    prompt: str,
    params: dict[str, float],
    variation_metadata: dict[str, Any],
) -> GeneratedMaterial:
    recipe = candidate_material_recipe(
        config.source,
        appearance_prompt=prompt,
        base_color=(
            params["base_color_r"],
            params["base_color_g"],
            params["base_color_b"],
        ),
        roughness=params["roughness"],
        metallic=params["metallic"],
    )
    source_reference: SourceMaterialReference | None = None
    if config.source.is_graph_backed:
        if config.source.source_usd is None or config.source.material_prim_path is None:
            raise MaterialRefinementError("source graph contract is incomplete")
        source_reference = SourceMaterialReference(
            usd_path=config.source.source_usd,
            material_prim_path=config.source.material_prim_path,
            usd_sha256=artifact_sha256(config.source.source_usd),
        )
    authored = author_material_package(
        MaterialAuthoringRequest(
            operation=(
                MaterialAuthoringOperation.MODIFY
                if source_reference is not None
                else MaterialAuthoringOperation.CREATE
            ),
            recipe=recipe,
            source=source_reference,
            textures=textures,
            material_profile=config.material_profile,
        ),
        package_dir,
        overwrite=True,
    )
    prototype_source: dict[str, Any] | None = None
    if source_reference is not None:
        prototype_source = {
            "library_path": source_reference.usd_path.as_posix(),
            "binding": source_reference.material_prim_path,
            "material_profile": authored.material_profile,
            "topology_sha256": authored.topology_sha256,
        }
    generated = GeneratedMaterial(
        recipe=recipe,
        textures=authored.textures,
        prototype_source=prototype_source,
    )
    write_generation_plan(
        package_dir / "material_generation_plan.yaml",
        MaterialGenerationPlan(
            materials=(recipe,),
            asset={"workflow": "material_refinement", **variation_metadata},
        ),
        (generated,),
    )
    return generated


def _generate_replica(
    *,
    config: MaterialRefinementConfig,
    generator: TextureVariationGenerator | None,
    source_swatch_path: Path,
    target_features: MaterialFeatures,
    params: dict[str, float],
    iteration: int,
    trial_index: int,
    replica_index: int,
    seed: int,
    prompt: str,
    diversity_features: tuple[MaterialFeatures, ...],
    minimum_diversity: float,
    diversity_weight: float,
    cancel_check: CancelCheck | None,
) -> ReplicaRecord:
    source_recipe = source_material_recipe(config.source)
    replica_dir = (
        config.output_dir
        / "attempts"
        / f"attempt_{iteration:03d}"
        / f"trial_{trial_index:04d}"
        / f"replica_{replica_index:02d}"
    )
    package_dir = replica_dir / "package"
    started = time.monotonic()
    try:
        _check_cancelled(cancel_check)
        trial_prompt = _trial_generation_prompt(prompt, params)
        base_color = (
            params["base_color_r"],
            params["base_color_g"],
            params["base_color_b"],
        )
        generator_name = generator.name if generator is not None else "scalar-pbr"
        publication_metadata = {
            "attempt": iteration,
            "trial_index": trial_index,
            "replica_index": replica_index,
            "seed": seed,
            "generator": generator_name,
            "source_representation": config.source.representation,
            "material_controls": {
                "base_color": list(base_color),
                "roughness": params["roughness"],
                "metallic": params["metallic"],
            },
        }
        artifacts = {
            "package_dir": package_dir.as_posix(),
            "material_usd": (package_dir / "material_library.usda").as_posix(),
            "materials_manifest": (package_dir / "materials.yaml").as_posix(),
            "generation_plan": (
                package_dir / "material_generation_plan.yaml"
            ).as_posix(),
        }
        artifact_metadata: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        metadata: dict[str, Any] = {
            "attempt": iteration,
            "trial_index": trial_index,
            "replica_index": replica_index,
            "generator": generator_name,
            "generation_prompt": trial_prompt,
            "source_representation": config.source.representation,
        }

        if config.source.textures is None:
            generated = _publish_candidate_package(
                config=config,
                package_dir=package_dir,
                textures=None,
                prompt=trial_prompt,
                params=params,
                variation_metadata=publication_metadata,
            )
            features = MaterialFeatures(
                base_color=base_color,
                roughness=params["roughness"],
                metallic=params["metallic"],
                color_source="authored_pbr",
            )
            metrics["generated_features"] = features.to_dict()
        else:
            if generator is None:
                raise MaterialRefinementDependencyError(
                    "textured refinement requires a Texture Variation generator"
                )
            service_artifacts_dir = replica_dir / "service_artifacts"
            variation = generator.generate(
                TextureVariationRequest(
                    source_asset_path=source_swatch_path,
                    material_path=source_recipe.binding,
                    prompt=trial_prompt,
                    reference_image_paths=(),
                    strength=params["strength"],
                    seed=seed,
                    variant_name=(
                        f"{source_recipe.material_id}_a{iteration:03d}_"
                        f"t{trial_index:04d}_r{replica_index:02d}"
                    ),
                ),
                output_dir=service_artifacts_dir,
                cancel_check=cancel_check,
            )
            _check_cancelled(cancel_check)
            service_textures = TextureMapSet(
                albedo=variation.albedo_path,
                normal=variation.normal_path,
                orm=variation.orm_path,
            )
            textures, service_validation, candidate_validation = (
                materialize_candidate_maps(
                    service_textures,
                    output_dir=(package_dir / "textures" / source_recipe.material_id),
                )
            )
            publication_metadata["strength"] = params["strength"]
            generated = _publish_candidate_package(
                config=config,
                package_dir=package_dir,
                textures=textures,
                prompt=trial_prompt,
                params=params,
                variation_metadata=publication_metadata,
            )
            if generated.textures is None:
                raise MaterialRefinementError(
                    "textured candidate publication lost its texture representation"
                )
            features = extract_material_features(
                albedo_path=generated.textures.albedo,
                orm_path=generated.textures.orm,
            )
            service_features = extract_material_features(
                albedo_path=service_textures.albedo,
                orm_path=service_textures.orm,
            )
            artifacts.update(
                {
                    "albedo": generated.textures.albedo.as_posix(),
                    "normal": generated.textures.normal.as_posix(),
                    "orm": generated.textures.orm.as_posix(),
                    "service_albedo": service_textures.albedo.as_posix(),
                    "service_normal": service_textures.normal.as_posix(),
                    "service_orm": service_textures.orm.as_posix(),
                }
            )
            artifact_metadata.update(
                {
                    channel: {
                        "media_type": "image/png",
                        "sha256": candidate_validation.sha256[channel],
                        "byte_size": candidate_validation.byte_size[channel],
                    }
                    for channel in ("albedo", "normal", "orm")
                }
            )
            artifact_metadata.update(
                {
                    f"service_{channel}": {
                        "media_type": "image/png",
                        "sha256": service_validation.sha256[channel],
                        "byte_size": service_validation.byte_size[channel],
                    }
                    for channel in ("albedo", "normal", "orm")
                }
            )
            metrics.update(
                {
                    "generated_features": features.to_dict(),
                    "service_features": service_features.to_dict(),
                    "service_map_validation": service_validation.to_dict(),
                    "candidate_map_validation": candidate_validation.to_dict(),
                }
            )
            metadata.update(
                {
                    "variant_asset_uri": redact_sensitive_config(
                        variation.variant_asset_uri, _path_context=True
                    ),
                    "variation_metadata": redact_sensitive_config(variation.metadata),
                    "variation_diagnostics": redact_sensitive_config(
                        list(variation.diagnostics)
                    ),
                }
            )

        objective = score_material_features(target_features, features, _PROXY_WEIGHTS)
        diversity_distances = [
            material_feature_distance(features, prior, _PROXY_WEIGHTS)
            for prior in diversity_features
        ]
        nearest_diversity = min(diversity_distances, default=None)
        diversity_penalty = (
            (1.0 - nearest_diversity) * diversity_weight
            if nearest_diversity is not None
            else 0.0
        )
        combined_objective = objective.value + diversity_penalty
        diversity_valid = (
            nearest_diversity is None or nearest_diversity >= minimum_diversity
        )
        artifact_metadata["material_usd"] = {
            "media_type": "model/vnd.usd",
            "sha256": artifact_sha256(package_dir / "material_library.usda"),
        }
        metrics.update(
            {
                "objective": objective.to_dict(),
                "combined_objective": combined_objective,
                "nearest_approved_distance": nearest_diversity,
                "diversity_penalty": diversity_penalty,
                "duration_seconds": time.monotonic() - started,
            }
        )
        return ReplicaRecord(
            seed=seed,
            objective_value=combined_objective if diversity_valid else None,
            success=diversity_valid,
            metrics=metrics,
            artifacts=artifacts,
            artifact_metadata=artifact_metadata,
            metadata=metadata,
            trial_dir=replica_dir.as_posix(),
            error=(
                None
                if diversity_valid
                else (
                    "candidate is below the minimum approved-material feature "
                    f"distance ({nearest_diversity:.6f} < {minimum_diversity:.6f})"
                )
            ),
        )
    except OptimizationCancelledError:
        raise
    except TextureVariationError as error:
        if cancel_check is not None and cancel_check():
            raise OptimizationCancelledError(str(error)) from error
        return _failed_replica(
            replica_dir, seed, started, error, iteration, trial_index
        )
    except Exception as error:
        return _failed_replica(
            replica_dir, seed, started, error, iteration, trial_index
        )


def _failed_replica(
    replica_dir: Path,
    seed: int,
    started: float,
    error: Exception,
    iteration: int,
    trial_index: int,
) -> ReplicaRecord:
    safe_error = _safe_error(error)
    error_path = _write_json(
        replica_dir / "error.json",
        {
            "error": safe_error,
            "error_type": type(error).__name__,
            "attempt": iteration,
            "trial_index": trial_index,
            "seed": seed,
        },
    )
    return ReplicaRecord(
        seed=seed,
        objective_value=None,
        success=False,
        metrics={"duration_seconds": time.monotonic() - started},
        artifacts={"error": error_path.as_posix()},
        trial_dir=replica_dir.as_posix(),
        error=safe_error,
    )


def _evaluate_candidate(
    *,
    config: MaterialRefinementConfig,
    generator: TextureVariationGenerator | None,
    source_swatch_path: Path,
    target_features: MaterialFeatures,
    settings: Any,
    params: dict[str, float],
    iteration: int,
    trial_index: int,
    trial_seed: int,
    prompt: str,
    diversity_features: tuple[MaterialFeatures, ...],
    minimum_diversity: float,
    diversity_weight: float,
    cancel_check: CancelCheck | None,
) -> TrialRecord:
    started = time.monotonic()
    replicas = [
        _generate_replica(
            config=config,
            generator=generator,
            source_swatch_path=source_swatch_path,
            target_features=target_features,
            params=params,
            iteration=iteration,
            trial_index=trial_index,
            replica_index=replica_index,
            seed=seed,
            prompt=prompt,
            diversity_features=diversity_features,
            minimum_diversity=minimum_diversity,
            diversity_weight=diversity_weight,
            cancel_check=cancel_check,
        )
        for replica_index, seed in enumerate(settings.replica_seeds)
    ]
    successful = [replica for replica in replicas if replica.success]
    all_succeeded = len(successful) == len(replicas)
    objective_value = (
        sum(cast(float, replica.objective_value) for replica in successful)
        / len(successful)
        if all_succeeded and successful
        else None
    )
    component_names = ("color", "roughness", "metallic")
    component_means = {
        name: sum(
            float(replica.metrics["objective"]["components"][name])
            for replica in successful
        )
        / len(successful)
        for name in component_names
        if successful
        and all(
            name in replica.metrics["objective"]["components"] for replica in successful
        )
    }
    errors = [replica.error for replica in replicas if replica.error]
    return TrialRecord(
        trial_index=trial_index,
        params=params,
        score=(
            _OBJECTIVE.optimizer_score(objective_value)
            if objective_value is not None
            else _OBJECTIVE.failure_penalty
        ),
        objective_value=objective_value,
        replicas=replicas,
        backend_metrics={
            "objective_name": _OBJECTIVE.name,
            "objective_unit": _OBJECTIVE.unit,
            "objective_components": component_means,
            "replica_count": len(replicas),
            "trial_seed": trial_seed,
        },
        duration_seconds=time.monotonic() - started,
        failed=not all_succeeded,
        error="; ".join(errors) if errors else None,
    )


def _target_candidate(
    target_features: MaterialFeatures,
    incumbent: dict[str, float],
) -> dict[str, float]:
    if (
        target_features.base_color is None
        or target_features.roughness is None
        or target_features.metallic is None
    ):
        raise MaterialRefinementError("material target is missing bounded PBR controls")
    candidate = dict(incumbent)
    candidate.update(
        {
            "base_color_r": target_features.base_color[0],
            "base_color_g": target_features.base_color[1],
            "base_color_b": target_features.base_color[2],
            "roughness": target_features.roughness,
            "metallic": target_features.metallic,
        }
    )
    return candidate


def _search_space_for_features(
    config: MaterialRefinementConfig,
    features: MaterialFeatures,
) -> MaterialSearchSpace:
    if (
        features.base_color is None
        or features.roughness is None
        or features.metallic is None
    ):
        raise MaterialRefinementError(
            "material features are missing bounded PBR controls"
        )
    return MaterialSearchSpace(
        config.search.parameters_for_values(
            base_color=features.base_color,
            roughness=features.roughness,
            metallic=features.metallic,
            variation=config.variation,
            include_texture_variation=config.source.textures is not None,
        )
    )


def _priority_first_runner(
    priority_candidates: tuple[dict[str, float], ...],
    optimizer_runner: Callable[..., None],
    *,
    priority_only: bool = False,
) -> Callable[..., None]:
    unique_candidates: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for candidate in priority_candidates:
        key = tuple(sorted((name, float(value)) for name, value in candidate.items()))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(dict(candidate))

    def run(
        search_space: MaterialSearchSpace,
        evaluate: Callable[[dict[str, float]], float],
        *,
        max_trials: int,
        seed: int,
        cancel_check: CancelCheck | None = None,
    ) -> None:
        evaluated = 0
        priority_limit = min(max_trials, 1 if priority_only else max_trials)
        for candidate in unique_candidates[:priority_limit]:
            if cancel_check is not None and cancel_check():
                return
            evaluate(dict(candidate))
            evaluated += 1
        remaining = max_trials - evaluated
        if (
            priority_only
            or remaining == 0
            or (cancel_check is not None and cancel_check())
        ):
            return
        optimizer_runner(
            search_space,
            evaluate,
            max_trials=remaining,
            seed=seed,
            cancel_check=cancel_check,
        )

    return run


def _candidate_features(representative: ReplicaRecord) -> MaterialFeatures:
    raw_features = representative.metrics.get("generated_features")
    if not isinstance(raw_features, dict):
        raise MaterialRefinementError("winning material is missing generated features")
    raw_color = raw_features.get("base_color")
    if not isinstance(raw_color, list | tuple) or len(raw_color) != 3:
        raise MaterialRefinementError(
            "winning material is missing generated base color"
        )
    return MaterialFeatures(
        base_color=(float(raw_color[0]), float(raw_color[1]), float(raw_color[2])),
        roughness=float(raw_features["roughness"]),
        metallic=float(raw_features["metallic"]),
        color_source=str(raw_features.get("color_source") or "judged_candidate"),
    )


def _judge_controls(config: MaterialRefinementConfig) -> tuple[str, ...]:
    controls = ["base_color", "roughness", "metallic"]
    if config.source.textures is not None:
        controls.append("texture_variation_content_and_strength")
    return tuple(controls)


def _representative_replica(trial: TrialRecord) -> ReplicaRecord:
    successful = [
        replica
        for replica in trial.replicas
        if replica.success and replica.objective_value is not None
    ]
    if not successful:
        raise MaterialRefinementError("best trial has no successful material package")
    return min(successful, key=lambda replica: cast(float, replica.objective_value))


def _publish_best_material(
    output_dir: Path, candidate: _JudgedCandidate
) -> tuple[Path, Path, dict[str, Path]]:
    source_package = Path(candidate.representative.artifacts["package_dir"])
    best_material_dir = output_dir / "best_material"
    shutil.copytree(source_package, best_material_dir)
    best_evidence_dir = output_dir / "best_evidence"
    shutil.copytree(candidate.evidence_dir, best_evidence_dir)
    artifacts: dict[str, Path] = {}
    for name, raw_path in candidate.representative.artifacts.items():
        path = Path(raw_path)
        if path.is_relative_to(source_package):
            path = best_material_dir / path.relative_to(source_package)
        artifacts[name] = path
    artifacts["package_dir"] = best_material_dir
    artifacts["evidence_dir"] = best_evidence_dir
    artifacts["render_manifest"] = best_evidence_dir / "render.json"
    artifacts["judge_verdict"] = best_evidence_dir / "judge.json"
    artifacts["swatch_usd"] = best_evidence_dir / "material_swatch.usda"
    for index, rendered_path in enumerate(
        candidate.render.rendered_image_paths, start=1
    ):
        artifacts[f"rendered_image_{index}"] = (
            best_evidence_dir / rendered_path.relative_to(candidate.evidence_dir)
        )
    return best_material_dir, best_evidence_dir, artifacts


def _select_terminal_candidate(candidates: list[_JudgedCandidate]) -> _JudgedCandidate:
    if not candidates:
        raise MaterialRefinementError("no visually judged material is available")
    finite = [
        candidate
        for candidate in candidates
        if candidate.trial.objective_value is not None
        and math.isfinite(candidate.trial.objective_value)
    ]
    if not finite:
        raise MaterialRefinementError("best material proxy score is not finite")
    approved = [candidate for candidate in finite if candidate.verdict.approved]
    pool = approved or finite
    return max(
        pool,
        key=lambda candidate: (
            candidate.verdict.score,
            -cast(float, candidate.trial.objective_value),
        ),
    )


def run_material_refinement(
    config: MaterialRefinementConfig,
    *,
    cancel_check: CancelCheck | None = None,
    on_trial: TrialCallback | None = None,
    generator: TextureVariationGenerator | None = None,
    render_task: RenderTaskLike | None = None,
    rendering_backend: Any | None = None,
    vlm_judge: VlmJudgeLike | None = None,
    diversity_features: tuple[MaterialFeatures, ...] = (),
    comparison_image_paths: tuple[Path, ...] = (),
    minimum_diversity: float = 0.0,
    diversity_weight: float = 0.0,
) -> MaterialRefinementResult:
    """Tune representation-preserving controls, then judge each sweep winner."""

    if not math.isfinite(minimum_diversity) or not 0.0 <= minimum_diversity <= 1.0:
        raise ValueError("minimum_diversity must be finite and in [0, 1]")
    if not math.isfinite(diversity_weight) or diversity_weight < 0.0:
        raise ValueError("diversity_weight must be finite and non-negative")
    if any(not path.is_file() for path in comparison_image_paths):
        raise FileNotFoundError("comparison material render does not exist")
    try:
        resolved_optimizer = resolve_optimizer(config.optimizer.name)
        optimizer_runner = get_runner(resolved_optimizer)
    except OptimizerUnavailableError as error:
        raise MaterialRefinementDependencyError(
            "requested optimizer is unavailable; install material-agent[refinement]"
        ) from error
    resolved_generator: TextureVariationGenerator | None = None
    if config.source.textures is not None:
        try:
            resolved_generator = generator or RestTextureVariationGenerator(
                config.variation
            )
        except Exception as error:
            raise MaterialRefinementDependencyError(
                f"Texture Variation generator is unavailable: {_safe_error(error)}"
            ) from error
    generator_name = (
        resolved_generator.name if resolved_generator is not None else "scalar-pbr"
    )
    if config.render.backend == "mock" and (
        rendering_backend is None or vlm_judge is None
    ):
        raise MaterialRefinementDependencyError(
            "render.backend='mock' requires injected rendering and judge fakes"
        )
    try:
        resolved_rendering_backend = rendering_backend or provision_rendering_backend(
            config.render
        )
        resolved_vlm_judge = vlm_judge or provision_vlm_judge(config.judge)
    except Exception as error:
        raise MaterialRefinementDependencyError(
            f"material visual dependency is unavailable: {_safe_error(error)}"
        ) from error
    try:
        _check_cancelled(cancel_check)
    except OptimizationCancelledError as error:
        raise MaterialRefinementCancelled(
            "material refinement was cancelled before startup"
        ) from error

    _prepare_output_dir(config)
    goal_evidence = config.goal.to_dict()
    goal_path = _write_json(config.output_dir / "goal.json", goal_evidence)
    target_inference_path = config.output_dir / "target_inference.json"
    summary_path = config.output_dir / "summary.json"
    try:
        source_material_dir, source_swatch_path, source_features = (
            _build_source_material(config)
        )
    except Exception as error:
        _write_json(
            config.output_dir / "source" / "error.json",
            {"error": _safe_error(error), "error_type": type(error).__name__},
        )
        _write_json(
            summary_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "success": False,
                "termination_reason": "source_material_failed",
                "goal_path": goal_path.as_posix(),
                "error": _safe_error(error),
                "error_type": type(error).__name__,
            },
        )
        raise MaterialRefinementDependencyError(
            f"source material preparation failed: {_safe_error(error)}"
        ) from error
    if (
        source_features.base_color is None
        or source_features.roughness is None
        or source_features.metallic is None
    ):
        raise MaterialRefinementError(
            "source material is missing measurable PBR features"
        )
    source_search_space = _search_space_for_features(config, source_features)
    controllable_properties = _judge_controls(config)
    try:
        _check_cancelled(cancel_check)
        target_inference = infer_material_target(
            appearance_prompt=config.goal.appearance_prompt,
            source_features=source_features,
            control_bounds=_semantic_pbr_control_bounds(),
            settings=config.judge,
            vlm_judge=resolved_vlm_judge,
            material_profile=config.material_profile,
            source_representation=config.source.representation,
        )
        _check_cancelled(cancel_check)
    except OptimizationCancelledError as error:
        _write_json(
            summary_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "success": False,
                "termination_reason": "cancelled",
                "goal_path": goal_path.as_posix(),
                "source_material_dir": source_material_dir.as_posix(),
                "source_swatch_path": source_swatch_path.as_posix(),
                "source_features": source_features.to_dict(),
                "search_space": source_search_space.to_dict(),
                "error": _safe_error(error),
                "error_type": type(error).__name__,
            },
        )
        raise MaterialRefinementCancelled(
            "material refinement was cancelled during target inference"
        ) from error
    except Exception as error:
        _write_json(
            summary_path,
            {
                "schema_version": _SCHEMA_VERSION,
                "success": False,
                "termination_reason": "target_inference_failed",
                "goal_path": goal_path.as_posix(),
                "source_material_dir": source_material_dir.as_posix(),
                "source_swatch_path": source_swatch_path.as_posix(),
                "source_features": source_features.to_dict(),
                "search_space": source_search_space.to_dict(),
                "error": _safe_error(error),
                "error_type": type(error).__name__,
            },
        )
        raise MaterialRefinementDependencyError(
            f"appearance target inference failed: {_safe_error(error)}"
        ) from error
    target_features = target_inference.target_features
    search_space = _merge_search_spaces(
        source_search_space,
        _search_space_for_features(config, target_features),
    )
    target_inference_evidence = {
        "inference": target_inference.to_dict(),
        "resolved_features": target_features.to_dict(),
        "source_anchors": source_features.to_dict(),
        "semantic_control_bounds": {
            name: {"min": bounds[0], "max": bounds[1]}
            for name, bounds in _semantic_pbr_control_bounds().items()
        },
        "source_search_space": source_search_space.to_dict(),
        "initial_search_space": search_space.to_dict(),
        "initial_expanded_controls": list(
            _changed_search_bounds(source_search_space, search_space)
        ),
        "revisions": [],
    }
    _write_json(target_inference_path, target_inference_evidence)

    history_path = config.output_dir / "history.jsonl"
    history_path.touch()
    initial_incumbent = {
        "base_color_r": source_features.base_color[0],
        "base_color_g": source_features.base_color[1],
        "base_color_b": source_features.base_color[2],
        "roughness": source_features.roughness,
        "metallic": source_features.metallic,
    }
    if config.source.textures is not None:
        initial_incumbent["strength"] = (
            config.variation.strength_min + config.variation.strength_max
        ) / 2
    attempts: list[MaterialRefinementAttempt] = []
    candidates: list[_JudgedCandidate] = []
    trial_count = 0

    def evaluate(
        iteration: RefinementIteration[_RefinementState],
    ) -> _SweepEvaluation:
        nonlocal trial_count
        state = iteration.state
        iteration_dir = (
            config.output_dir / "attempts" / f"attempt_{iteration.iteration:03d}"
        )
        iteration_dir.mkdir(parents=True, exist_ok=True)
        settings = _iteration_settings(config, iteration.iteration)
        _write_json(
            iteration_dir / "search.json",
            {
                "attempt": iteration.iteration,
                "optimizer": resolved_optimizer,
                "optimizer_seed": settings.seed,
                "max_trials": settings.max_trials,
                "incumbent": state.incumbent,
                "target_features": state.target_features.to_dict(),
                "target_inference": state.target_inference.to_dict(),
                "search_space": state.search_space.to_dict(),
                "prompt": state.prompt,
                "previous_feedback": state.previous_feedback,
                "generator": generator_name,
                "source_representation": config.source.representation,
            },
        )

        def evaluate_trial(
            params: dict[str, float], trial_index: int, trial_seed: int
        ) -> TrialRecord:
            return _evaluate_candidate(
                config=config,
                generator=resolved_generator,
                source_swatch_path=source_swatch_path,
                target_features=state.target_features,
                settings=settings,
                params=params,
                iteration=iteration.iteration,
                trial_index=trial_index,
                trial_seed=trial_seed,
                prompt=state.prompt,
                diversity_features=diversity_features,
                minimum_diversity=minimum_diversity,
                diversity_weight=diversity_weight,
                cancel_check=cancel_check,
            )

        def record_trial(trial: TrialRecord) -> None:
            nonlocal trial_count
            trial_count += 1
            _append_json_line(
                history_path, {"attempt": iteration.iteration, **trial.to_dict()}
            )
            if on_trial is not None:
                on_trial(iteration.iteration, trial)

        tuning = run_tuning(
            settings=settings,
            search_space=state.search_space,
            evaluate_trial=evaluate_trial,
            resolve_optimizer=lambda _name: resolved_optimizer,
            get_optimizer_runner=lambda _name: _priority_first_runner(
                (
                    _target_candidate(state.target_features, state.incumbent),
                    state.incumbent,
                ),
                optimizer_runner,
                priority_only=(
                    config.source.textures is None and not diversity_features
                ),
            ),
            cancel_check=cancel_check,
            on_trial=record_trial,
        )
        best = tuning.best()
        representative = _representative_replica(best) if best is not None else None
        return _SweepEvaluation(
            iteration=iteration.iteration,
            state=state,
            tuning=tuning,
            best_trial=best,
            representative=representative,
            iteration_dir=iteration_dir,
        )

    def judge(
        iteration: RefinementIteration[_RefinementState],
        evaluation: _SweepEvaluation,
    ) -> _SweepJudgment:
        if evaluation.tuning.cancelled:
            return _SweepJudgment(cancelled=True)
        if evaluation.best_trial is None or evaluation.representative is None:
            return _SweepJudgment(error="no successful trials")
        evidence_dir = evaluation.iteration_dir / "evidence"
        try:
            _check_cancelled(cancel_check)
            render = render_material_swatch(
                material_usd_path=Path(
                    evaluation.representative.artifacts["material_usd"]
                ),
                material_binding=source_material_recipe(config.source).binding,
                evidence_dir=evidence_dir,
                settings=config.render,
                rendering_backend=resolved_rendering_backend,
                render_task=render_task,
            )
            _write_json(
                evidence_dir / "render.json",
                {
                    "attempt": iteration.iteration,
                    "winning_trial_index": evaluation.best_trial.trial_index,
                    "aggregate_proxy_score": evaluation.best_trial.objective_value,
                    "representative_proxy_score": (
                        evaluation.representative.objective_value
                    ),
                    "representative_seed": evaluation.representative.seed,
                    **render.to_dict(),
                },
            )
            _check_cancelled(cancel_check)
            verdict = judge_rendered_material(
                target=config.goal,
                render_evidence=render,
                settings=config.judge,
                vlm_judge=resolved_vlm_judge,
                iteration=iteration.iteration,
                previous_feedback=evaluation.state.previous_feedback,
                comparison_image_paths=comparison_image_paths,
                material_profile=config.material_profile,
                source_representation=config.source.representation,
                controllable_properties=controllable_properties,
            )
            _write_json(
                evidence_dir / "judge.json",
                {"attempt": iteration.iteration, **verdict.to_dict()},
            )
            candidates.append(
                _JudgedCandidate(
                    iteration=iteration.iteration,
                    trial=evaluation.best_trial,
                    representative=evaluation.representative,
                    render=render,
                    verdict=verdict,
                    target_features=evaluation.state.target_features,
                    search_space=evaluation.state.search_space,
                    evidence_dir=evidence_dir,
                )
            )
            target_revision: MaterialTargetInference | None = None
            revised_search_space: MaterialSearchSpace | None = None
            target_revision_error: str | None = None
            if not verdict.approved and not iteration.is_last:
                candidate_features = _candidate_features(evaluation.representative)
                try:
                    target_revision = infer_material_target(
                        appearance_prompt=config.goal.appearance_prompt,
                        source_features=candidate_features,
                        control_bounds=_semantic_pbr_control_bounds(),
                        settings=config.judge,
                        vlm_judge=resolved_vlm_judge,
                        previous_feedback=verdict.feedback,
                        material_profile=config.material_profile,
                        source_representation=config.source.representation,
                    )
                    revised_search_space = _merge_search_spaces(
                        evaluation.state.search_space,
                        _search_space_for_features(
                            config, target_revision.target_features
                        ),
                    )
                except OptimizationCancelledError:
                    raise
                except Exception as error:
                    target_revision_error = _safe_error(error)
                    revision_evidence = {
                        "after_attempt": iteration.iteration,
                        "previous_target_features": (
                            evaluation.state.target_features.to_dict()
                        ),
                        "candidate_features": candidate_features.to_dict(),
                        "previous_search_space": (
                            evaluation.state.search_space.to_dict()
                        ),
                        "error": target_revision_error,
                        "error_type": type(error).__name__,
                    }
                else:
                    revision_evidence = {
                        "after_attempt": iteration.iteration,
                        "previous_target_features": (
                            evaluation.state.target_features.to_dict()
                        ),
                        "candidate_features": candidate_features.to_dict(),
                        "previous_search_space": (
                            evaluation.state.search_space.to_dict()
                        ),
                        "revised_search_space": revised_search_space.to_dict(),
                        "expanded_controls": list(
                            _changed_search_bounds(
                                evaluation.state.search_space,
                                revised_search_space,
                            )
                        ),
                        "inference": target_revision.to_dict(),
                    }
                    target_inference_evidence["resolved_features"] = (
                        target_revision.target_features.to_dict()
                    )
                    target_inference_evidence["latest_inference"] = (
                        target_revision.to_dict()
                    )
                revisions = cast(
                    list[dict[str, Any]], target_inference_evidence["revisions"]
                )
                revisions.append(revision_evidence)
                _write_json(evidence_dir / "target_revision.json", revision_evidence)
                _write_json(target_inference_path, target_inference_evidence)
            _check_cancelled(cancel_check)
            return _SweepJudgment(
                render=render,
                verdict=verdict,
                target_revision=target_revision,
                revised_search_space=revised_search_space,
                evidence_dir=evidence_dir,
                target_revision_error=target_revision_error,
            )
        except OptimizationCancelledError:
            return _SweepJudgment(evidence_dir=evidence_dir, cancelled=True)
        except Exception as error:
            safe_error = _safe_error(error)
            _write_json(
                evidence_dir / "error.json",
                {"error": safe_error, "error_type": type(error).__name__},
            )
            return _SweepJudgment(evidence_dir=evidence_dir, error=safe_error)

    def decide(
        _iteration: RefinementIteration[_RefinementState],
        evaluation: _SweepEvaluation,
        judgment: _SweepJudgment,
    ) -> RefinementDecision:
        if evaluation.tuning.cancelled or judgment.cancelled:
            return RefinementDecision.stop("cancelled")
        if evaluation.best_trial is None:
            return RefinementDecision.stop("no_successful_trials")
        if judgment.error is not None:
            return RefinementDecision.stop("visual_evaluation_failed")
        if judgment.verdict is not None and judgment.verdict.approved:
            return RefinementDecision.approve()
        if judgment.target_revision_error is not None:
            return RefinementDecision.stop("target_revision_failed")
        return RefinementDecision.continue_()

    def revise(
        iteration: RefinementIteration[_RefinementState],
        evaluation: _SweepEvaluation,
        judgment: _SweepJudgment,
    ) -> _RefinementState:
        if (
            evaluation.best_trial is None
            or judgment.verdict is None
            or judgment.target_revision is None
            or judgment.revised_search_space is None
        ):
            raise AssertionError("continuing refinement requires a judged winner")
        return _RefinementState(
            prompt=_candidate_prompt(
                _goal_generation_prompt(config),
                iteration=iteration.iteration + 1,
                feedback=judgment.verdict.feedback,
            ),
            incumbent=dict(evaluation.best_trial.params),
            target_features=judgment.target_revision.target_features,
            target_inference=judgment.target_revision,
            search_space=judgment.revised_search_space,
            previous_feedback=judgment.verdict.feedback,
        )

    def record_attempt(record: Any) -> None:
        evaluation: _SweepEvaluation = record.evaluation
        judgment: _SweepJudgment = record.judgment
        best = evaluation.best_trial
        representative = evaluation.representative
        result_path = evaluation.iteration_dir / "result.json"
        attempt = MaterialRefinementAttempt(
            iteration=evaluation.iteration,
            optimizer_seed=_iteration_settings(config, evaluation.iteration).seed,
            prompt=evaluation.state.prompt,
            target_features=evaluation.state.target_features,
            search_space=evaluation.state.search_space,
            trial_count=len(evaluation.tuning.history),
            best_params=dict(best.params) if best is not None else {},
            best_score=best.objective_value if best is not None else None,
            representative_score=(
                representative.objective_value if representative is not None else None
            ),
            representative_seed=(
                representative.seed if representative is not None else None
            ),
            approved=bool(judgment.verdict and judgment.verdict.approved),
            cancelled=evaluation.tuning.cancelled or judgment.cancelled,
            render_evidence=judgment.render,
            judge_verdict=judgment.verdict,
            target_revision=judgment.target_revision,
            revised_search_space=judgment.revised_search_space,
            error=judgment.error or judgment.target_revision_error,
            result_path=result_path,
        )
        attempts.append(attempt)
        _write_json(result_path, attempt.to_dict())

    refinement = run_refinement(
        initial_state=_RefinementState(
            prompt=_goal_generation_prompt(config),
            incumbent=initial_incumbent,
            target_features=target_features,
            target_inference=target_inference,
            search_space=search_space,
        ),
        max_iterations=config.max_refinements + 1,
        evaluate=evaluate,
        judge=judge,
        decide=decide,
        revise=revise,
        on_iteration=record_attempt,
    )
    base_summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "success": bool(candidates),
        "termination_reason": refinement.termination_reason,
        "approved": refinement.approved,
        "optimizer": resolved_optimizer,
        "generator": generator_name,
        "source_representation": config.source.representation,
        "proxy_objective": {
            "name": _OBJECTIVE.name,
            "unit": _OBJECTIVE.unit,
            "direction": _OBJECTIVE.direction,
            "approval_authority": False,
        },
        "visual_judge": {
            "approval_authority": True,
            "score_threshold": config.judge.score_threshold,
            "provider": config.judge.provider_evidence(),
        },
        "goal": goal_evidence,
        "target_inference_path": target_inference_path.as_posix(),
        "target_inference": target_inference_evidence,
        "source_material_dir": source_material_dir.as_posix(),
        "source_swatch_path": source_swatch_path.as_posix(),
        "source_features": source_features.to_dict(),
        "initial_search_space": search_space.to_dict(),
        "search_space": (
            refinement.final_state.search_space.to_dict()
            if refinement.final_state is not None
            else search_space.to_dict()
        ),
        "trial_count": trial_count,
        "attempts": [attempt.to_dict() for attempt in attempts],
    }
    if not candidates:
        _write_json(summary_path, base_summary)
        if refinement.termination_reason == "cancelled":
            raise MaterialRefinementCancelled(
                "material refinement was cancelled before visual judgment"
            )
        terminal_error = next(
            (attempt.error for attempt in reversed(attempts) if attempt.error), None
        )
        if terminal_error:
            raise MaterialRefinementError(
                f"material visual evaluation failed: {terminal_error}"
            )
        raise MaterialRefinementError("material refinement produced no judged material")

    terminal = _select_terminal_candidate(candidates)
    if terminal.trial.objective_value is None or not math.isfinite(
        terminal.trial.objective_value
    ):
        raise MaterialRefinementError("best material proxy score is not finite")
    if terminal.representative.objective_value is None or not math.isfinite(
        terminal.representative.objective_value
    ):
        raise MaterialRefinementError("representative proxy score is not finite")
    best_material_dir, best_evidence_dir, best_artifacts = _publish_best_material(
        config.output_dir, terminal
    )
    result = MaterialRefinementResult(
        termination_reason=refinement.termination_reason,
        optimizer=resolved_optimizer,
        best_params=dict(terminal.trial.params),
        best_score=terminal.trial.objective_value,
        representative_score=terminal.representative.objective_value,
        representative_seed=terminal.representative.seed,
        judge_score=terminal.verdict.score,
        judge_decision=terminal.verdict.decision,
        best_material_dir=best_material_dir,
        best_evidence_dir=best_evidence_dir,
        best_artifacts=best_artifacts,
        target_features=terminal.target_features,
        search_space=terminal.search_space,
        source_material_dir=source_material_dir,
        source_swatch_path=source_swatch_path,
        history_path=history_path,
        summary_path=summary_path,
        attempts=tuple(attempts),
        trial_count=trial_count,
    )
    _write_json(summary_path, {**base_summary, **result.to_dict()})
    return result


__all__ = [
    "MaterialRefinementAttempt",
    "MaterialRefinementCancelled",
    "MaterialRefinementDependencyError",
    "MaterialRefinementError",
    "MaterialRefinementResult",
    "MaterialSearchSpace",
    "run_material_refinement",
]
