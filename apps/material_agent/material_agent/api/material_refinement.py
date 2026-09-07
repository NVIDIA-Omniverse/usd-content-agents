# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public Python API for rendered material refinement and variation sets."""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.config import clone_config_containers
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import normalize_required_config
from material_agent.api.types import APIResult
from material_agent.material_refinement import (
    MaterialRefinementCancelled,
    MaterialRefinementConfig,
    MaterialRefinementDependencyError,
    MaterialVariationConfig,
    run_material_refinement,
    run_material_variations,
)

logger = logging.getLogger(__name__)
_REFINEMENT_FAILURE = "Material refinement failed"
_VARIATION_FAILURE = "Material variation failed"
_LEGACY_REFINEMENT_WARNING = (
    "The rendered material-refinement controller is a compatibility API. New "
    "agentic and fixed workflows should plan and judge externally, then publish "
    "through MaterialAuthoringRequest and author_material_package."
)
_LEGACY_VARIATION_WARNING = (
    "The rendered material-variation controller is a compatibility API. New "
    "agentic and fixed workflows should plan and judge externally, then publish "
    "through MaterialAuthoringRequest and author_material_package."
)


@dataclass
class MaterialRefinementInput:
    """Configuration and optional runtime injections for one material."""

    config: Path | dict[str, Any]
    output_dir_override: Path | None = None
    config_path: Path | None = None
    cancel_checker: Callable[[], bool] | None = field(default=None, repr=False)
    generator: Any | None = field(default=None, repr=False)
    render_task: Any | None = field(default=None, repr=False)
    rendering_backend: Any | None = field(default=None, repr=False)
    vlm_judge: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.config = normalize_required_config(self.config)
        if self.output_dir_override is not None:
            self.output_dir_override = Path(self.output_dir_override)
        if self.config_path is not None:
            self.config_path = Path(self.config_path)


@dataclass(repr=False)
class MaterialRefinementOutput(SecretSafeReprMixin, APIResult):
    """Stable public projection of a material-refinement result."""

    approved: bool = False
    termination_reason: str = "unknown"
    optimizer: str | None = None
    best_score: float | None = None
    judge_score: float | None = None
    best_material_dir: Path | None = None
    best_artifacts: dict[str, Path] = field(default_factory=dict)
    summary_path: Path | None = None
    trial_count: int = 0
    attempt_count: int = 0
    cancelled: bool = False
    diagnostic: str | None = None


@dataclass(repr=False)
class MaterialVariationOutput(SecretSafeReprMixin, APIResult):
    """Stable public projection of a material-variation-set result."""

    status: str = "unknown"
    termination_reason: str = "unknown"
    selected_count: int = 0
    manifest_path: Path | None = None
    material_library_path: Path | None = None
    materials_manifest_path: Path | None = None
    selected_render_paths: list[Path] = field(default_factory=list)
    cancelled: bool = False
    diagnostic: str | None = None


def _load_mapping(
    config: Path | dict[str, Any], config_path: Path | None
) -> tuple[dict[str, Any], Path]:
    if isinstance(config, dict):
        data = clone_config_containers(config)
        base_dir = config_path.parent.resolve() if config_path else Path.cwd().resolve()
        return data, base_dir
    resolved = Path(config).resolve()
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("material refinement configuration must be a mapping")
    return loaded, resolved.parent


async def arun_material_refinement(
    params: MaterialRefinementInput,
) -> MaterialRefinementOutput:
    """Run one material refinement without blocking an async caller."""

    try:
        warnings.warn(_LEGACY_REFINEMENT_WARNING, DeprecationWarning, stacklevel=2)
        data, base_dir = _load_mapping(params.config, params.config_path)
        config = MaterialRefinementConfig.from_mapping(
            data,
            base_dir=base_dir,
            output_dir_override=params.output_dir_override,
        )
        result = await asyncio.to_thread(
            run_material_refinement,
            config,
            cancel_check=params.cancel_checker,
            generator=params.generator,
            render_task=params.render_task,
            rendering_backend=params.rendering_backend,
            vlm_judge=params.vlm_judge,
        )
        cancelled = result.termination_reason == "cancelled"
        success = result.success
        return MaterialRefinementOutput(
            success=success,
            error=None if success else _REFINEMENT_FAILURE,
            approved=result.approved,
            termination_reason=result.termination_reason,
            optimizer=result.optimizer,
            best_score=result.best_score,
            judge_score=result.judge_score,
            best_material_dir=result.best_material_dir,
            best_artifacts=dict(result.best_artifacts),
            summary_path=result.summary_path,
            trial_count=result.trial_count,
            attempt_count=len(result.attempts),
            cancelled=cancelled,
            diagnostic=(
                "Material refinement was cancelled"
                if cancelled
                else (
                    None
                    if success
                    else (f"Material refinement stopped: {result.termination_reason}")
                )
            ),
        )
    except MaterialRefinementCancelled:
        return MaterialRefinementOutput(
            success=False,
            error=_REFINEMENT_FAILURE,
            termination_reason="cancelled",
            cancelled=True,
            diagnostic="Material refinement was cancelled",
        )
    except MaterialRefinementDependencyError as error:
        logger.exception(_REFINEMENT_FAILURE)
        return MaterialRefinementOutput(
            success=False, error=_REFINEMENT_FAILURE, diagnostic=str(error)
        )
    except Exception:
        logger.exception(_REFINEMENT_FAILURE)
        return MaterialRefinementOutput(
            success=False,
            error=_REFINEMENT_FAILURE,
            diagnostic="Material refinement failed; inspect local logs for details",
        )


async def arun_material_variations(
    params: MaterialRefinementInput,
) -> MaterialVariationOutput:
    """Run a material variation set without blocking an async caller."""

    try:
        warnings.warn(_LEGACY_VARIATION_WARNING, DeprecationWarning, stacklevel=2)
        data, base_dir = _load_mapping(params.config, params.config_path)
        config = MaterialVariationConfig.from_mapping(
            data,
            base_dir=base_dir,
            output_dir_override=params.output_dir_override,
        )
        result = await asyncio.to_thread(
            run_material_variations,
            config,
            cancel_check=params.cancel_checker,
            generator=params.generator,
            render_task=params.render_task,
            rendering_backend=params.rendering_backend,
            vlm_judge=params.vlm_judge,
        )
        return MaterialVariationOutput(
            success=result.success,
            error=None if result.success else _VARIATION_FAILURE,
            status=result.status,
            termination_reason=result.termination_reason,
            selected_count=len(result.selected_materials),
            manifest_path=result.manifest_path,
            material_library_path=result.material_library_path,
            materials_manifest_path=result.materials_manifest_path,
            selected_render_paths=list(result.selected_render_paths),
            cancelled=result.status == "cancelled",
            diagnostic=None if result.success else "Variation set is incomplete",
        )
    except Exception:
        logger.exception(_VARIATION_FAILURE)
        return MaterialVariationOutput(
            success=False,
            error=_VARIATION_FAILURE,
            diagnostic="Material variation failed; inspect local logs for details",
        )


def run_material_refinement_api(
    params: MaterialRefinementInput,
) -> MaterialRefinementOutput:
    """Run one material refinement synchronously."""

    return asyncio.run(arun_material_refinement(params))


def run_material_variations_api(
    params: MaterialRefinementInput,
) -> MaterialVariationOutput:
    """Run a material variation set synchronously."""

    return asyncio.run(arun_material_variations(params))


def refine_material(
    config: Path | dict[str, Any], *, output_dir: Path | None = None
) -> MaterialRefinementOutput:
    """Convenience API for one rendered material refinement."""

    return run_material_refinement_api(
        MaterialRefinementInput(config=config, output_dir_override=output_dir)
    )


def create_material_variations(
    config: Path | dict[str, Any], *, output_dir: Path | None = None
) -> MaterialVariationOutput:
    """Convenience API for a rendered material variation set."""

    return run_material_variations_api(
        MaterialRefinementInput(config=config, output_dir_override=output_dir)
    )


__all__ = [
    "MaterialRefinementInput",
    "MaterialRefinementOutput",
    "MaterialVariationOutput",
    "arun_material_refinement",
    "arun_material_variations",
    "create_material_variations",
    "refine_material",
    "run_material_refinement_api",
    "run_material_variations_api",
]
