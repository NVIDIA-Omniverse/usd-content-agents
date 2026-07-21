# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict API for Joint Agent.

This module provides the programmatic API for running VLM predictions.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    is_model_authentication_error,
    public_model_failure_message,
)

from joint_agent.api.types import APIResult

logger = logging.getLogger(__name__)

_PREDICT_FAILURE_MESSAGE = "Prediction failed"


@dataclass
class PredictInput:
    """Input parameters for prediction API."""

    config: Path | dict[str, Any]
    """Path to config file or config dictionary"""

    dataset_override: Path | None = None
    """Override dataset path from config"""

    output_dir_override: Path | None = None
    """Override output directory from config"""

    resume: bool = False
    """Resume from existing predictions"""

    stream_predictions: bool = True
    """Stream predictions to file as they are produced"""

    verbose: bool = False
    """Enable verbose logging"""


@dataclass
class PredictOutput(APIResult):
    """Output from prediction API."""

    predictions_path: Path | None = None
    """Path to predictions file"""

    predictions_count: int = 0
    """Number of predictions made"""

    failed_count: int = 0
    """Number of failed predictions"""

    token_stats: dict[str, Any] = field(default_factory=dict)
    """Token usage statistics"""


def run_predict(params: PredictInput) -> PredictOutput:
    """Run VLM predictions on a dataset.

    This is the main entry point for running predictions programmatically.
    It creates and executes the prediction workflow.

    Args:
        params: Prediction input parameters

    Returns:
        PredictOutput with results

    Example:
        >>> from joint_agent.api import PredictInput, run_predict
        >>> params = PredictInput(config=Path("config.yaml"))
        >>> result = run_predict(params)
        >>> print(f"Predictions: {result.predictions_count}")
    """
    return asyncio.run(arun_predict(params))


async def arun_predict(params: PredictInput) -> PredictOutput:
    """Async version of run_predict.

    Args:
        params: Prediction input parameters

    Returns:
        PredictOutput with results
    """
    try:
        from joint_agent.workflows import create_prediction_workflow_from_config

        # Create workflow
        workflow = create_prediction_workflow_from_config()

        # Prepare context
        context: dict[str, Any] = {
            "resume": params.resume,
            "stream_predictions": params.stream_predictions,
            "verbose": params.verbose,
        }

        # Handle config (path or dict)
        if isinstance(params.config, dict):
            context["config_dict"] = params.config
        else:
            context["config_path"] = str(params.config)

        # Add overrides if provided
        if params.dataset_override:
            context["dataset_override"] = str(params.dataset_override)
        if params.output_dir_override:
            context["output_dir_override"] = str(params.output_dir_override)

        # Run workflow
        result = workflow.run(context)

        # Check for errors
        if result.get("error") or result.get("workflow_terminated"):
            failure_message = (
                MODEL_AUTHENTICATION_FAILURE_MESSAGE
                if is_model_authentication_error(result.get("error"))
                else _PREDICT_FAILURE_MESSAGE
            )
            return PredictOutput(
                success=False,
                error=failure_message,
            )

        # Extract results
        predictions_path = result.get("predictions_path")
        predictions_count = result.get("predictions_count", 0)
        failed_count = result.get("failed_count", 0)
        token_stats = result.get("token_stats", {})

        return PredictOutput(
            success=True,
            predictions_path=Path(predictions_path) if predictions_path else None,
            predictions_count=predictions_count,
            failed_count=failed_count,
            token_stats=token_stats,
        )

    except Exception as error:
        failure_message = public_model_failure_message(error, _PREDICT_FAILURE_MESSAGE)
        logger.error(failure_message)
        return PredictOutput(success=False, error=failure_message)
