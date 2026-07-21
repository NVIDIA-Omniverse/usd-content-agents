# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refine API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import clone_config_containers
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
    retain_safe_result_text,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import (
    diagnostic_path,
    normalize_required_config,
)
from material_agent.api.types import APIResult

logger = logging.getLogger(__name__)

_REFINE_FAILURE_MESSAGE = "Material refinement failed"


def _projected_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _projected_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


@dataclass
class RefineInput:
    """Input parameters for refine API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        config_path: Optional source path used to anchor relative paths for dict config
        max_iterations_override: Override maximum iterations from config
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    max_iterations_override: int | None = None
    verbose: bool = False
    config_path: Path | None = None

    def __post_init__(self):
        """Validate inputs."""
        self.config = normalize_required_config(self.config)
        if self.config_path is not None:
            self.config_path = Path(self.config_path)


@dataclass
class IterationResult:
    """Result from a single iteration."""

    iteration: int
    judge_score: float | None
    continue_iteration: bool
    materials_applied_count: int = 0
    prims_with_materials: int = 0


@dataclass(repr=False)
class RefineOutput(SecretSafeReprMixin, APIResult):
    """Output results from refine API."""

    iteration_count: int = 0
    final_output_path: Path | None = None
    final_judge_score: float | None = None
    termination_reason: str = "unknown"
    iteration_results: list[IterationResult] = field(default_factory=list)
    all_iteration_outputs: list[Path] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


async def arun_refine(params: RefineInput) -> RefineOutput:
    """Refine materials on USD with VLM-based iterative refinement.

    This command executes a predict-apply-judge loop repeatedly until the judge
    approves the results or maximum iterations is reached. It uses VLM to predict
    materials, applies them to USD, renders the result, and has a VLM judge evaluate
    quality by comparing against reference images.

    The configuration file must specify:
    - dataset: Path to the dataset JSONL file
    - input_usd_path: Path to the input USD file
    - output_usd_path: Path for the final output (optional)
    - iteration: Iteration settings (max_iterations, save_intermediate, etc.)
    - judge: Judge configuration (reference_images, vlm settings, etc.)

    Args:
        params: Refine input parameters

    Returns:
        RefineOutput with results or error information
    """
    logger.info("Starting material refinement via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info("Configuration file: %s", diagnostic_path(params.config))

    if params.max_iterations_override:
        logger.info(f"Max iterations override: {params.max_iterations_override}")

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_iterative_apply_workflow_from_config,
        )

        # Create workflow
        logger.info("Creating iterative apply workflow...")
        workflow = create_iterative_apply_workflow_from_config()

        # Prepare initial context with config and overrides
        initial_context: dict[str, Any] = {
            "max_iterations_override": params.max_iterations_override,
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = clone_config_containers(params.config)
            if params.config_path is not None:
                initial_context["config_path"] = str(params.config_path)
        else:
            initial_context["config_path"] = str(params.config)

        # Run the workflow
        logger.info("Running material refinement with iterative loop...")
        result = await workflow.arun(initial_context=initial_context)
        safe_result = project_result_metadata(result)

        # Check if workflow was successful
        if result.get("iteration_count", 0) > 0:
            iteration_count = _projected_int(safe_result, "iteration_count")
            iteration_results_raw = safe_result.get("iteration_results", [])
            if not isinstance(iteration_results_raw, list):
                iteration_results_raw = []
            final_iteration = safe_result.get("final_iteration", {})
            if not isinstance(final_iteration, dict):
                final_iteration = {}
            termination_reason = (
                retain_safe_result_text(safe_result.get("termination_reason"))
                or "unknown"
            )
            runtime_outputs = result.get("all_iteration_outputs", [])
            all_outputs = (
                [
                    safe_path
                    for value in runtime_outputs
                    if (safe_path := retain_safe_result_path(value)) is not None
                ]
                if isinstance(runtime_outputs, list)
                else []
            )
            final_output_path = retain_safe_result_path(result.get("final_output_path"))

            # Convert iteration results to structured format
            iteration_results = [
                IterationResult(
                    iteration=_projected_int(item, "iteration"),
                    judge_score=_projected_float(item, "judge_score"),
                    continue_iteration=(
                        item.get("continue_iteration")
                        if isinstance(item.get("continue_iteration"), bool)
                        else False
                    ),
                    materials_applied_count=_projected_int(
                        item, "materials_applied_count"
                    ),
                    prims_with_materials=_projected_int(item, "prims_with_materials"),
                )
                for item in iteration_results_raw
                if isinstance(item, dict)
            ]

            logger.info(
                f"Material refinement completed after {iteration_count} iterations"
            )

            return RefineOutput(
                success=True,
                iteration_count=iteration_count,
                final_output_path=final_output_path,
                final_judge_score=_projected_float(final_iteration, "judge_score"),
                termination_reason=termination_reason,
                iteration_results=iteration_results,
                all_iteration_outputs=all_outputs,
                raw_result=safe_result,
            )
        else:
            error_msg = "Material refinement workflow did not complete successfully"
            logger.error(error_msg)
            return RefineOutput(
                success=False,
                error=error_msg,
                raw_result=safe_result,
            )

    except Exception:
        logger.error(_REFINE_FAILURE_MESSAGE)
        return RefineOutput(
            success=False,
            error=_REFINE_FAILURE_MESSAGE,
        )


def run_refine(params: RefineInput) -> RefineOutput:
    """Run iterative refinement synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Refine input parameters

    Returns:
        RefineOutput with results or error information
    """
    return asyncio.run(arun_refine(params))


async def arefine(
    config: Path | dict[str, Any],
    max_iterations_override: int | None = None,
    verbose: bool = False,
) -> RefineOutput:
    """Async convenience function for refine API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        max_iterations_override: Override max iterations from config
        verbose: Enable verbose output

    Returns:
        RefineOutput with results
    """
    params = RefineInput(
        config=config,
        max_iterations_override=max_iterations_override,
        verbose=verbose,
    )
    return await arun_refine(params)


def refine(
    config: Path | dict[str, Any],
    max_iterations_override: int | None = None,
    verbose: bool = False,
) -> RefineOutput:
    """Sync convenience function for refine API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        max_iterations_override: Override max iterations from config
        verbose: Enable verbose output

    Returns:
        RefineOutput with results
    """
    return asyncio.run(arefine(config, max_iterations_override, verbose))
