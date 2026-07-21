# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_understanding.utils.credentials import path_exists_with_safe_diagnostics
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import diagnostic_path, normalize_required_config
from material_agent.api.types import APIResult, MetricsResult

logger = logging.getLogger(__name__)

_EVALUATE_FAILURE_MESSAGE = "Evaluation failed"
_PREDICTIONS_FILE_NOT_FOUND_MESSAGE = "Predictions file not found"


@dataclass
class EvaluateInput:
    """Input parameters for evaluate API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        config_path: Optional source path used to anchor relative paths for dict config
        predictions_override: Optional path to override predictions from config
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    predictions_override: Path | None = None
    verbose: bool = False
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate inputs."""
        self.config = normalize_required_config(self.config)
        if self.config_path is not None:
            self.config_path = Path(self.config_path)

        if self.predictions_override:
            self.predictions_override = Path(self.predictions_override)
            if not path_exists_with_safe_diagnostics(
                self.predictions_override,
                label="predictions file",
            ):
                raise FileNotFoundError(_PREDICTIONS_FILE_NOT_FOUND_MESSAGE)


@dataclass(repr=False)
class EvaluateOutput(SecretSafeReprMixin, APIResult):
    """Output results from evaluate API."""

    metrics: MetricsResult | None = None
    evaluation_path: Path | None = None
    html_report_path: Path | None = None
    raw_result: dict[str, Any] | None = None


async def arun_evaluate(params: EvaluateInput) -> EvaluateOutput:
    """Evaluate existing predictions using an LLM judge.

    This command loads an evaluation configuration file and evaluates predictions
    against ground truth using the configured LLM judge. It calculates
    metrics including Functional Correctness Score (FCS) and success rate.

    Args:
        params: Evaluate input parameters

    Returns:
        EvaluateOutput with results or error information
    """
    logger.info("Starting evaluate via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info("Configuration file: %s", diagnostic_path(params.config))

    if params.predictions_override:
        logger.info(
            "Predictions override: %s",
            diagnostic_path(params.predictions_override),
        )

    try:
        # Import workflow factory
        from material_agent.workflows import create_evaluation_workflow_from_config

        # Create config-driven evaluation workflow
        logger.info("Creating config-driven evaluation workflow...")
        workflow = create_evaluation_workflow_from_config()

        # Prepare initial context
        initial_context: dict[str, Any] = {
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
            if params.config_path is not None:
                initial_context["config_path"] = str(params.config_path)
        else:
            initial_context["config_path"] = str(params.config)

        # Add predictions override if provided
        if params.predictions_override:
            initial_context["predictions_path"] = str(params.predictions_override)

        # Run the evaluation workflow
        logger.info("Running evaluation...")
        result = await workflow.arun(initial_context=initial_context)

        # Check if evaluation was successful
        if result.get("evaluation_complete"):
            safe_result = project_result_metadata(result)
            metrics_dict = safe_result.get("metrics", {})
            if not isinstance(metrics_dict, dict):
                metrics_dict = {}
            evaluation_path = retain_safe_result_path(result.get("evaluation_path"))
            html_report_path = retain_safe_result_path(result.get("html_report_path"))

            logger.info("Evaluation completed successfully")

            metrics = MetricsResult.from_projected_dict(metrics_dict)

            return EvaluateOutput(
                success=True,
                metrics=metrics,
                evaluation_path=evaluation_path,
                html_report_path=html_report_path,
                raw_result=safe_result,
            )
        else:
            logger.error(_EVALUATE_FAILURE_MESSAGE)
            return EvaluateOutput(
                success=False,
                error=_EVALUATE_FAILURE_MESSAGE,
            )

    except Exception:
        logger.error(_EVALUATE_FAILURE_MESSAGE)
        return EvaluateOutput(
            success=False,
            error=_EVALUATE_FAILURE_MESSAGE,
        )


def run_evaluate(params: EvaluateInput) -> EvaluateOutput:
    """Evaluate existing predictions synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Evaluate input parameters

    Returns:
        EvaluateOutput with results or error information
    """
    return asyncio.run(arun_evaluate(params))


async def aevaluate(
    config: Path | dict[str, Any],
    predictions_override: Path | None = None,
    verbose: bool = False,
) -> EvaluateOutput:
    """Async convenience function for evaluate API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        predictions_override: Optional path to override predictions from config
        verbose: Enable verbose output

    Returns:
        EvaluateOutput with results
    """
    params = EvaluateInput(
        config=config,
        predictions_override=predictions_override,
        verbose=verbose,
    )
    return await arun_evaluate(params)


def evaluate(
    config: Path | dict[str, Any],
    predictions_override: Path | None = None,
    verbose: bool = False,
) -> EvaluateOutput:
    """Sync convenience function for evaluate API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        predictions_override: Optional path to override predictions from config
        verbose: Enable verbose output

    Returns:
        EvaluateOutput with results
    """
    return asyncio.run(aevaluate(config, predictions_override, verbose))
