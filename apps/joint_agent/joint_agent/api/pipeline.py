# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API for Joint Agent.

This module provides the programmatic API for running multi-step pipelines.

Usage patterns:

1. **Full Input class** - Maximum control:
    ```python
    from joint_agent.api import PipelineInput, run_pipeline

    params = PipelineInput(config=Path("config.yaml"))
    result = run_pipeline(params)
    ```

2. **Convenience function** - Minimal usage:
    ```python
    from joint_agent.api import pipeline

    result = pipeline(Path("config.yaml"))
    ```
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import clone_config_containers
from world_understanding.agentic.events import EventListener
from world_understanding.utils.credentials import (
    path_is_file_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
)
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    is_model_authentication_error,
    public_model_failure_message,
)
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_path,
    retain_safe_result_text,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

logger = logging.getLogger(__name__)

_PIPELINE_FAILURE_MESSAGE = "Pipeline execution failed"
_DRY_RUN_FAILURE_MESSAGE = "Unable to build pipeline execution plan"


@dataclass
class PipelineInput:
    """Input parameters for pipeline API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        source_config_path: Optional original YAML path used only to anchor
            relative paths when config is an in-memory override
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        session_id: Optional session ID to reuse existing session directory
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        verbose: Enable verbose output
        event_listener: Optional event listener for progress reporting
        cancel_checker: Optional callback returning True when the run should stop
    """

    config: Path | dict[str, Any]
    skip_steps: list[str] = field(default_factory=list)
    only_steps: list[str] = field(default_factory=list)
    session_id: str | None = None
    resume: bool = False
    dry_run: bool = False
    clean: bool = False
    verbose: bool = False
    event_listener: EventListener | None = None
    cancel_checker: Callable[[], bool] | None = None
    source_config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate inputs."""
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
            if self.source_config_path is not None:
                self.source_config_path = Path(self.source_config_path)
                if not path_is_file_with_safe_diagnostics(
                    self.source_config_path,
                    label="pipeline source config file",
                ):
                    raise FileNotFoundError(
                        "Source config file not found: "
                        f"{redact_sensitive_path(self.source_config_path)}"
                    )
        else:
            self.config = Path(self.config)
            if not path_is_file_with_safe_diagnostics(
                self.config,
                label="pipeline configuration file",
            ):
                raise FileNotFoundError(
                    f"Config file not found: {redact_sensitive_path(self.config)}"
                )


@dataclass(repr=False)
class PipelineOutput(SecretSafeReprMixin):
    """Output results from pipeline API."""

    success: bool
    error: str | None = None
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    session_id: str | None = None
    working_dir: Path | None = None
    raw_result: dict[str, Any] | None = None


async def arun_pipeline(params: PipelineInput) -> PipelineOutput:
    """Execute a multi-step joint agent pipeline asynchronously.

    This is the core async implementation. The sync version delegates to this.

    A typical pipeline includes:
    1. optimize_usd: Optimize USD for rendering (optional)
    2. build_dataset_usd: Render USD prims to images
    3. build_dataset_prepare_dataset: Prepare dataset with prompts
    4. predict: Run VLM inference

    The pipeline automatically connects outputs from one step to inputs of the next.

    Args:
        params: Pipeline input parameters

    Returns:
        PipelineOutput with results or error information
    """
    # Get or create event listener
    listener = params.event_listener
    if listener is None:
        from world_understanding.agentic.events import create_default_listener

        listener = create_default_listener(verbose=params.verbose)

    safe_skip_steps = redact_sensitive_config(params.skip_steps)
    if not isinstance(safe_skip_steps, list):
        safe_skip_steps = []
    safe_only_steps = redact_sensitive_config(params.only_steps)
    if not isinstance(safe_only_steps, list):
        safe_only_steps = []

    # Emit workflow started event
    listener.event(
        "workflow.started",
        {
            "workflow_type": "pipeline",
            "config_type": "dict" if isinstance(params.config, dict) else "file",
            "skip_steps": safe_skip_steps,
            "only_steps": safe_only_steps,
        },
    )

    listener.info("Starting pipeline via API")
    if isinstance(params.config, dict):
        listener.info("Using in-memory config dictionary")
    else:
        listener.info(f"Configuration file: {redact_sensitive_path(params.config)}")

    if params.skip_steps:
        listener.info(f"Skip steps: {', '.join(safe_skip_steps)}")
    if params.only_steps:
        listener.info(f"Only steps: {', '.join(safe_only_steps)}")
    if params.resume:
        listener.info("Resume mode enabled")
    if params.clean:
        listener.info("Clean mode enabled (will delete working dir and output files)")

    if params.dry_run:
        listener.info("Dry run mode - showing execution plan only")
        return _dry_run_pipeline(params)

    if params.cancel_checker and params.cancel_checker():
        listener.event(
            "workflow.cancelled",
            {
                "workflow_type": "pipeline",
                "message": "Pipeline cancellation requested",
            },
        )
        raise asyncio.CancelledError("Pipeline cancellation requested")

    try:
        from joint_agent.workflows import create_unified_pipeline_workflow

        listener.info("Creating unified pipeline workflow")
        workflow = create_unified_pipeline_workflow()

        # Prepare initial context
        initial_context: dict[str, Any] = {
            "skip_steps": params.skip_steps,
            "only_steps": params.only_steps,
            "resume": params.resume,
            "clean": params.clean,
            "verbose": params.verbose,
            "event_listener": listener,
        }
        if params.cancel_checker is not None:
            initial_context["cancel_checker"] = params.cancel_checker

        # Add session_id if provided
        if params.session_id:
            initial_context["session_id"] = params.session_id
            safe_session_id = redact_sensitive_config(
                params.session_id,
                _path_context=True,
            )
            listener.info(f"Using provided session ID: {safe_session_id}")

        # Add config as either path or dict
        if isinstance(params.config, dict):
            initial_context["config_dict"] = clone_config_containers(params.config)
            if params.source_config_path is not None:
                initial_context["config_path"] = str(params.source_config_path)
        else:
            initial_context["config_path"] = str(params.config)

        listener.info("Running unified pipeline workflow")
        listener.event("workflow.executing", {"workflow_type": "pipeline"})

        result = await workflow.arun(initial_context)

        # Check if workflow returned nothing
        if not result:
            error_msg = "Pipeline workflow did not complete successfully"
            listener.error(error_msg)
            listener.event(
                "workflow.failed",
                {"workflow_type": "pipeline", "error": error_msg},
            )
            return PipelineOutput(
                success=False,
                error=error_msg,
                skipped_steps=safe_skip_steps,
                raw_result=result,
            )

        # Check for workflow errors even if result exists
        if result.get("error") or result.get("workflow_terminated"):
            failure_message = (
                MODEL_AUTHENTICATION_FAILURE_MESSAGE
                if is_model_authentication_error(result.get("error"))
                else _PIPELINE_FAILURE_MESSAGE
            )
            failed_task = result.get("failed_task", "unknown")
            safe_failed_task = (
                retain_safe_result_text(
                    failed_task,
                    path_context=True,
                )
                or "<redacted>"
            )
            listener.error(
                f"Pipeline failed at task '{safe_failed_task}': {failure_message}"
            )
            listener.event(
                "workflow.failed",
                {
                    "workflow_type": "pipeline",
                    "error": failure_message,
                    "failed_task": safe_failed_task,
                },
            )
            # Still extract partial results if available
            pipeline_results = result.get("pipeline_results", {})
            safe_pipeline_results = project_result_metadata(pipeline_results)
            completed_steps = list(safe_pipeline_results.keys())
            raw_session_id = result.get("session_id")
            safe_session_id = retain_safe_result_text(
                raw_session_id,
                path_context=True,
            )
            raw_working_dir = result.get("working_dir")
            safe_working_dir = retain_safe_result_path(raw_working_dir)
            return PipelineOutput(
                success=False,
                error=failure_message,
                step_results=safe_pipeline_results,
                completed_steps=completed_steps,
                skipped_steps=safe_skip_steps,
                session_id=safe_session_id,
                working_dir=safe_working_dir,
            )

        # Pipeline succeeded. Runtime context remains raw through execution;
        # public result metadata is a detached, credential-safe projection.
        safe_result = project_result_metadata(result)
        safe_pipeline_results = safe_result.get("pipeline_results", {})
        if not isinstance(safe_pipeline_results, dict):
            safe_pipeline_results = {}
        completed_steps = list(safe_pipeline_results.keys())
        session_id = retain_safe_result_text(
            result.get("session_id"),
            path_context=True,
        )
        working_dir = retain_safe_result_path(result.get("working_dir"))

        # Emit completion event
        listener.event(
            "workflow.completed",
            {
                "workflow_type": "pipeline",
                "completed_steps": completed_steps,
            },
        )
        listener.info("Pipeline completed successfully")

        return PipelineOutput(
            success=True,
            step_results=safe_pipeline_results,
            completed_steps=completed_steps,
            skipped_steps=safe_skip_steps,
            session_id=session_id,
            working_dir=working_dir,
            raw_result=safe_result,
        )

    except Exception as error:
        failure_message = public_model_failure_message(error, _PIPELINE_FAILURE_MESSAGE)
        listener.error(failure_message)
        listener.event(
            "workflow.failed",
            {"workflow_type": "pipeline", "error": failure_message},
        )
        return PipelineOutput(
            success=False,
            error=failure_message,
        )


def _dry_run_pipeline(params: PipelineInput) -> PipelineOutput:
    """Perform a dry run of the pipeline.

    Args:
        params: Pipeline input parameters

    Returns:
        PipelineOutput with execution plan information
    """
    import yaml

    try:
        # Load config
        if isinstance(params.config, dict):
            pipeline_config = params.config
        else:
            with open(params.config, encoding="utf-8") as f:
                pipeline_config = yaml.safe_load(f)

        from joint_agent.api.defaults import PIPELINE_STEP_NAMES

        # Detect unified config format
        steps_section = pipeline_config.get("steps", pipeline_config)

        planned_steps = []
        skipped_steps = []

        for step in PIPELINE_STEP_NAMES:
            if step not in steps_section:
                continue

            step_config = steps_section[step]

            # Check if enabled
            enabled = (
                step_config.get("enabled") if isinstance(step_config, dict) else True
            )
            if enabled is None:
                has_config = (
                    any(k != "enabled" for k in step_config.keys())
                    if isinstance(step_config, dict)
                    else bool(step_config)
                )
                enabled = has_config
            if not enabled:
                continue

            if params.skip_steps and step in params.skip_steps:
                skipped_steps.append(step)
            elif params.only_steps and step not in params.only_steps:
                skipped_steps.append(step)
            else:
                planned_steps.append(step)

        logger.info(f"Dry run - would execute steps: {planned_steps}")
        logger.info(f"Dry run - would skip steps: {skipped_steps}")

        return PipelineOutput(
            success=True,
            completed_steps=planned_steps,
            skipped_steps=skipped_steps,
        )

    except Exception:
        logger.error(_DRY_RUN_FAILURE_MESSAGE)
        return PipelineOutput(
            success=False,
            error=_DRY_RUN_FAILURE_MESSAGE,
        )


def run_pipeline(params: PipelineInput) -> PipelineOutput:
    """Execute a multi-step joint agent pipeline synchronously.

    This is a wrapper around the async implementation.

    Args:
        params: Pipeline input parameters

    Returns:
        PipelineOutput with results or error information
    """
    return asyncio.run(arun_pipeline(params))


async def apipeline(
    config: Path | dict[str, Any],
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    session_id: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    clean: bool = False,
    event_listener: EventListener | None = None,
    verbose: bool = False,
    cancel_checker: Callable[[], bool] | None = None,
    source_config_path: Path | None = None,
) -> PipelineOutput:
    """Async convenience function for pipeline API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        session_id: Optional session ID to reuse existing session directory
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output
        cancel_checker: Optional callback returning True when the run should stop
        source_config_path: Optional original YAML path used to anchor relative paths
            when config is an in-memory dictionary

    Returns:
        PipelineOutput with results
    """
    params = PipelineInput(
        config=config,
        skip_steps=skip_steps or [],
        only_steps=only_steps or [],
        session_id=session_id,
        resume=resume,
        dry_run=dry_run,
        clean=clean,
        verbose=verbose,
        event_listener=event_listener,
        cancel_checker=cancel_checker,
        source_config_path=source_config_path,
    )
    return await arun_pipeline(params)


def pipeline(
    config: Path | dict[str, Any],
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    session_id: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    clean: bool = False,
    event_listener: EventListener | None = None,
    verbose: bool = False,
    cancel_checker: Callable[[], bool] | None = None,
    source_config_path: Path | None = None,
) -> PipelineOutput:
    """Sync convenience function for pipeline API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        session_id: Optional session ID to reuse existing session directory
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output
        cancel_checker: Optional callback returning True when the run should stop
        source_config_path: Optional original YAML path used to anchor relative paths
            when config is an in-memory dictionary

    Returns:
        PipelineOutput with results
    """
    return asyncio.run(
        apipeline(
            config=config,
            skip_steps=skip_steps,
            only_steps=only_steps,
            session_id=session_id,
            resume=resume,
            dry_run=dry_run,
            clean=clean,
            event_listener=event_listener,
            verbose=verbose,
            cancel_checker=cancel_checker,
            source_config_path=source_config_path,
        )
    )
