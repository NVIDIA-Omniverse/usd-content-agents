# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark API for Material Agent."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import EventListener
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import diagnostic_path, normalize_required_config
from material_agent.api.types import APIResult, MetricsResult

logger = logging.getLogger(__name__)

_BENCHMARK_FAILURE_MESSAGE = "Benchmark failed"


def _metrics_from_projected_result(
    projected_result: dict[str, Any],
) -> MetricsResult | None:
    """Build typed metrics only from the detached public projection."""
    metrics_data = projected_result.get("metrics")
    if not isinstance(metrics_data, dict) or not metrics_data:
        return None

    return MetricsResult.from_projected_dict(metrics_data)


@dataclass
class BenchmarkInput:
    """Input parameters for benchmark API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        config_path: Optional source path used to anchor relative paths for dict config
        dataset_override: Optional path to override dataset from config
        output_dir_override: Optional path to override output directory from config
        resume: Resume from existing predictions.jsonl
        stream_predictions: Append predictions as they are produced
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output
    """

    config: Path | dict[str, Any]
    dataset_override: Path | None = None
    output_dir_override: Path | None = None
    resume: bool = False
    stream_predictions: bool = True
    event_listener: EventListener | None = None
    verbose: bool = False
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate inputs."""
        self.config = normalize_required_config(self.config)
        if self.config_path is not None:
            self.config_path = Path(self.config_path)

        if self.dataset_override:
            self.dataset_override = Path(self.dataset_override)

        if self.output_dir_override:
            self.output_dir_override = Path(self.output_dir_override)


@dataclass(repr=False)
class BenchmarkOutput(SecretSafeReprMixin, APIResult):
    """Output results from benchmark API."""

    metrics: MetricsResult | None = None
    evaluation_path: Path | None = None
    predictions_path: Path | None = None
    raw_result: dict[str, Any] | None = None


async def arun_benchmark(params: BenchmarkInput) -> BenchmarkOutput:
    """Run benchmark evaluation asynchronously.

    This is the core async implementation. The sync version delegates to this.

    Args:
        params: Benchmark input parameters

    Returns:
        BenchmarkOutput with results or error information
    """
    # Get or create event listener
    listener = params.event_listener
    if listener is None:
        from world_understanding.agentic.events import create_default_listener

        listener = create_default_listener(verbose=params.verbose)

    # Emit workflow started event
    listener.event(
        "workflow.started",
        {
            "workflow_type": "benchmark",
            "config_type": "dict" if isinstance(params.config, dict) else "file",
        },
    )

    listener.info("Starting benchmark via API")
    if isinstance(params.config, dict):
        listener.info("Using in-memory config dictionary")
    else:
        listener.info(f"Configuration file: {diagnostic_path(params.config)}")

    if params.dataset_override:
        listener.info(f"Dataset override: {diagnostic_path(params.dataset_override)}")
    if params.output_dir_override:
        listener.info(
            f"Output directory override: {diagnostic_path(params.output_dir_override)}"
        )

    try:
        # Import workflow factory
        from material_agent.workflows.factory import (
            create_benchmark_workflow_from_config,
        )

        # Apply defaults if using dict config
        config_to_use = params.config
        if isinstance(params.config, dict):
            from material_agent.api.defaults import get_benchmark_config_with_defaults

            config_to_use = get_benchmark_config_with_defaults(params.config)
            logger.info("Applied default values to config dictionary")

        # Build initial context
        initial_context: dict[str, Any] = {
            "dataset_override": str(params.dataset_override)
            if params.dataset_override
            else None,
            "output_dir_override": str(params.output_dir_override)
            if params.output_dir_override
            else None,
            "resume": params.resume,
            "stream_predictions": params.stream_predictions,
            "verbose": params.verbose,
        }

        # Add config as either path or dict
        if isinstance(config_to_use, dict):
            initial_context["config_dict"] = config_to_use
            if params.config_path is not None:
                initial_context["config_path"] = str(params.config_path)
        else:
            initial_context["config_path"] = str(config_to_use)

        # Create workflow
        listener.info("Creating config-driven benchmark workflow")
        workflow = create_benchmark_workflow_from_config()

        # Run the benchmark workflow asynchronously
        listener.info("Running benchmark workflow...")
        listener.event("workflow.executing", {"workflow_type": "benchmark"})

        result = await workflow.arun(initial_context)
        safe_result = project_result_metadata(result)

        # Runtime context/results stay raw while the workflow runs. Build every
        # public metric and metadata field from a detached projection.
        metrics = _metrics_from_projected_result(safe_result)

        if metrics is not None:
            evaluation_path = retain_safe_result_path(
                result.get("evaluation_path") if isinstance(result, dict) else None
            )
            predictions_path = retain_safe_result_path(
                result.get("predictions_path") if isinstance(result, dict) else None
            )
            safe_event_metrics = project_result_metadata(
                {"metrics": metrics.to_dict()}
            ).get("metrics", {})

            # Emit completion event
            listener.event(
                "workflow.completed",
                {
                    "workflow_type": "benchmark",
                    "metrics": safe_event_metrics,
                    "evaluation_path": (
                        diagnostic_path(evaluation_path) if evaluation_path else None
                    ),
                    "predictions_path": (
                        diagnostic_path(predictions_path) if predictions_path else None
                    ),
                },
            )
            listener.info("Benchmark completed successfully")

            return BenchmarkOutput(
                success=True,
                metrics=metrics,
                evaluation_path=evaluation_path,
                predictions_path=predictions_path,
                raw_result=safe_result,
            )
        else:
            safe_result["error"] = _BENCHMARK_FAILURE_MESSAGE
            listener.error(_BENCHMARK_FAILURE_MESSAGE)
            listener.event(
                "workflow.failed",
                {
                    "workflow_type": "benchmark",
                    "error": _BENCHMARK_FAILURE_MESSAGE,
                },
            )
            return BenchmarkOutput(
                success=False,
                error=_BENCHMARK_FAILURE_MESSAGE,
                raw_result=safe_result,
            )

    except Exception:
        listener.error(_BENCHMARK_FAILURE_MESSAGE)
        listener.event(
            "workflow.failed",
            {
                "workflow_type": "benchmark",
                "error": _BENCHMARK_FAILURE_MESSAGE,
            },
        )
        return BenchmarkOutput(
            success=False,
            error=_BENCHMARK_FAILURE_MESSAGE,
        )


def run_benchmark(params: BenchmarkInput) -> BenchmarkOutput:
    """Run benchmark evaluation synchronously.

    This is a wrapper around the async implementation for backward compatibility.

    Args:
        params: Benchmark input parameters

    Returns:
        BenchmarkOutput with results or error information
    """
    return asyncio.run(arun_benchmark(params))


async def abenchmark(
    config: Path | dict[str, Any],
    dataset_override: Path | None = None,
    output_dir_override: Path | None = None,
    resume: bool = False,
    stream_predictions: bool = True,
    event_listener: EventListener | None = None,
    verbose: bool = False,
) -> BenchmarkOutput:
    """Async convenience function for benchmark API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        dataset_override: Optional path to override dataset from config
        output_dir_override: Optional path to override output directory from config
        resume: Resume from existing predictions.jsonl
        stream_predictions: Append predictions as they are produced
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output

    Returns:
        BenchmarkOutput with results
    """
    params = BenchmarkInput(
        config=config,
        dataset_override=dataset_override,
        output_dir_override=output_dir_override,
        resume=resume,
        stream_predictions=stream_predictions,
        event_listener=event_listener,
        verbose=verbose,
    )
    return await arun_benchmark(params)


def benchmark(
    config: Path | dict[str, Any],
    dataset_override: Path | None = None,
    output_dir_override: Path | None = None,
    resume: bool = False,
    stream_predictions: bool = True,
    event_listener: EventListener | None = None,
    verbose: bool = False,
) -> BenchmarkOutput:
    """Sync convenience function for benchmark API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        dataset_override: Optional path to override dataset from config
        output_dir_override: Optional path to override output directory from config
        resume: Resume from existing predictions.jsonl
        stream_predictions: Append predictions as they are produced
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output

    Returns:
        BenchmarkOutput with results
    """
    return asyncio.run(
        abenchmark(
            config,
            dataset_override,
            output_dir_override,
            resume,
            stream_predictions,
            event_listener,
            verbose,
        )
    )
