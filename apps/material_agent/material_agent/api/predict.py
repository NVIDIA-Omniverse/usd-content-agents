# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    is_model_authentication_error,
    public_model_failure_message,
)
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import diagnostic_path, normalize_required_config
from material_agent.api.types import APIResult, projected_int

logger = logging.getLogger(__name__)

_PREDICT_FAILURE_MESSAGE = "Predict failed"


@dataclass
class PredictInput:
    """Input parameters for predict API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        resume: Resume from last checkpoint
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    resume: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate inputs."""
        self.config = normalize_required_config(self.config)


@dataclass(repr=False)
class PredictOutput(SecretSafeReprMixin, APIResult):
    """Output results from predict API."""

    predictions_path: Path | None = None
    report_path: Path | None = None
    num_predictions: int = 0
    raw_result: dict[str, Any] | None = None


async def arun_predict(params: PredictInput) -> PredictOutput:
    """Run material predictions on a dataset without evaluation asynchronously.

    This is the core async implementation. The sync version delegates to this.
    This is equivalent to: pipeline --only predict

    Args:
        params: Predict input parameters

    Returns:
        PredictOutput with results or error information
    """
    logger.info("Starting predict via API")
    if isinstance(params.config, dict):
        logger.info("Using in-memory config dictionary")
    else:
        logger.info("Configuration file: %s", diagnostic_path(params.config))

    try:
        # Apply defaults if using dict config
        config_to_use = params.config
        if isinstance(params.config, dict):
            from material_agent.api.defaults import get_predict_config_with_defaults

            config_to_use = get_predict_config_with_defaults(params.config)
            logger.info("Applied default values to config dictionary")

        # Import the pipeline API to reuse logic
        from material_agent.api.pipeline import PipelineInput, arun_pipeline

        # Create pipeline params with only=predict
        pipeline_params = PipelineInput(
            config=config_to_use,
            skip_steps=[],
            only_steps=["predict"],
            resume=params.resume,
            dry_run=False,
            clean=False,
            verbose=params.verbose,
        )

        # Run pipeline asynchronously
        pipeline_result = await arun_pipeline(pipeline_params)

        if pipeline_result.success:
            # Extract predict-specific results
            predict_result = pipeline_result.step_results.get("predict", {})
            safe_predict_result = project_result_metadata(predict_result)

            return PredictOutput(
                success=True,
                predictions_path=retain_safe_result_path(
                    predict_result.get("predictions_path")
                ),
                report_path=retain_safe_result_path(predict_result.get("report_path")),
                num_predictions=projected_int(
                    safe_predict_result.get("num_predictions")
                ),
                raw_result=safe_predict_result,
            )
        else:
            failure_message = (
                MODEL_AUTHENTICATION_FAILURE_MESSAGE
                if is_model_authentication_error(pipeline_result.error)
                else _PREDICT_FAILURE_MESSAGE
            )
            logger.error(failure_message)
            return PredictOutput(
                success=False,
                error=failure_message,
            )

    except Exception as error:
        failure_message = public_model_failure_message(error, _PREDICT_FAILURE_MESSAGE)
        logger.error(failure_message)
        return PredictOutput(
            success=False,
            error=failure_message,
        )


def run_predict(params: PredictInput) -> PredictOutput:
    """Run material predictions synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Predict input parameters

    Returns:
        PredictOutput with results or error information
    """
    return asyncio.run(arun_predict(params))


async def apredict(
    config: Path | dict[str, Any],
    resume: bool = False,
    verbose: bool = False,
) -> PredictOutput:
    """Async convenience function for predict API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        resume: Resume from last checkpoint
        verbose: Enable verbose output

    Returns:
        PredictOutput with results
    """
    params = PredictInput(
        config=config,
        resume=resume,
        verbose=verbose,
    )
    return await arun_predict(params)


def predict(
    config: Path | dict[str, Any],
    resume: bool = False,
    verbose: bool = False,
) -> PredictOutput:
    """Sync convenience function for predict API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        resume: Resume from last checkpoint
        verbose: Enable verbose output

    Returns:
        PredictOutput with results
    """
    return asyncio.run(apredict(config, resume, verbose))
