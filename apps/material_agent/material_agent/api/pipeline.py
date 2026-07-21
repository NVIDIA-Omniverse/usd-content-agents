# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API for Material Agent."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import EventListener
from world_understanding.utils.credentials import redact_sensitive_config
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    is_model_authentication_error,
    public_model_failure_message,
)
from world_understanding.utils.result_projection import (
    project_result_metadata,
    retain_safe_result_text,
)
from world_understanding.utils.safe_repr import SecretSafeReprMixin

from material_agent.api.diagnostics import (
    diagnostic_path,
    normalize_required_config,
)
from material_agent.api.types import APIResult

logger = logging.getLogger(__name__)

_PIPELINE_FAILURE_MESSAGE = "Pipeline execution failed"
_DRY_RUN_FAILURE_MESSAGE = "Unable to build pipeline execution plan"


@dataclass
class PipelineInput:
    """Input parameters for pipeline API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        config_path: Optional source path used to anchor relative paths for dict config
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output
        session_id: Optional session ID to reuse existing session directory
        cancel_checker: Optional callback returning True when the run should stop
    """

    config: Path | dict[str, Any]
    skip_steps: list[str] = field(default_factory=list)
    only_steps: list[str] = field(default_factory=list)
    resume: bool = False
    dry_run: bool = False
    clean: bool = False
    event_listener: EventListener | None = None
    verbose: bool = False
    session_id: str | None = None
    simulate: bool = False
    cancel_checker: Callable[[], bool] | None = None
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate inputs."""
        self.config = normalize_required_config(self.config)
        if self.config_path is not None:
            self.config_path = Path(self.config_path)


@dataclass(repr=False)
class PipelineOutput(SecretSafeReprMixin, APIResult):
    """Output results from pipeline API."""

    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


async def arun_pipeline(params: PipelineInput) -> PipelineOutput:
    """Execute a multi-step material agent pipeline asynchronously.

    This is the core async implementation. The sync version delegates to this.

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir, input.usd_path, and output.usd_path.

    A typical pipeline includes:
    1. build_dataset_usd: Build dataset from USD files
    2. build_dataset_pdf_vectorstore: Build vector store from PDFs (optional)
    3. build_dataset_prepare_dataset: Prepare dataset with specifications
    4. predict/benchmark: Run VLM inference
    5. apply: Apply predicted materials to USD

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
        listener.info(f"Configuration file: {diagnostic_path(params.config)}")

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

    # --- Simulate mode: patch all backends to "mock" ---
    simulate_config_dict: dict[str, Any] | None = None
    simulate_config_path: Path | None = None
    try:
        if params.simulate:
            import yaml

            from material_agent.api.simulate_config import patch_config_for_simulate

            if isinstance(params.config, dict):
                raw_dict = params.config
                simulate_config_path = params.config_path
            else:
                with open(params.config, encoding="utf-8") as f:
                    raw_dict = yaml.safe_load(f)
                # Keep the original path so the path resolver can resolve
                # relative paths (usd_path, materials, etc.) from the config dir.
                simulate_config_path = params.config
            simulate_config_dict = patch_config_for_simulate(raw_dict)
            listener.info("Simulate mode: all backends patched to 'mock'")

        # Import workflow factory
        from material_agent.workflows import create_unified_pipeline_workflow

        listener.info("Creating unified pipeline workflow")
        workflow = create_unified_pipeline_workflow()

        # Prepare initial context
        initial_context: dict[str, Any] = {
            "skip_steps": params.skip_steps,
            "only_steps": params.only_steps,
            "resume": params.resume,
            "clean": params.clean,
            "event_listener": listener,  # Pass listener to workflow
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
        if simulate_config_dict is not None:
            initial_context["config_dict"] = simulate_config_dict
            # Preserve the original config_path so the path resolver can
            # resolve relative paths from the config file's directory.
            if simulate_config_path is not None:
                initial_context["config_path"] = str(simulate_config_path)
        elif isinstance(params.config, dict):
            initial_context["config_dict"] = params.config
            if params.config_path is not None:
                initial_context["config_path"] = str(params.config_path)
        else:
            initial_context["config_path"] = str(params.config)

        listener.info("Running unified pipeline workflow")
        listener.event("workflow.executing", {"workflow_type": "pipeline"})

        result = await workflow.arun(initial_context)

        # Check if workflow encountered errors
        if not result:
            error_msg = "Pipeline workflow did not complete successfully"
            listener.error(error_msg)
            listener.event(
                "workflow.failed", {"workflow_type": "pipeline", "error": error_msg}
            )
            return PipelineOutput(
                success=False,
                error=error_msg,
                step_results={},
                completed_steps=[],
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
            safe_result = project_result_metadata(result)
            safe_result["error"] = failure_message
            safe_result["failed_task"] = safe_failed_task
            safe_pipeline_results = safe_result.get("pipeline_results", {})
            if not isinstance(safe_pipeline_results, dict):
                safe_pipeline_results = {}
            completed_steps = list(safe_pipeline_results.keys())
            return PipelineOutput(
                success=False,
                error=failure_message,
                step_results=safe_pipeline_results,
                completed_steps=completed_steps,
                skipped_steps=safe_skip_steps,
                raw_result=safe_result,
            )

        # Pipeline succeeded. Keep the workflow context raw while tasks execute,
        # then publish only a detached diagnostic projection of its results.
        safe_result = project_result_metadata(result)
        safe_pipeline_results = safe_result.get("pipeline_results", {})
        if not isinstance(safe_pipeline_results, dict):
            safe_pipeline_results = {}
        completed_steps = list(safe_pipeline_results.keys())

        # Emit completion event
        listener.event(
            "workflow.completed",
            {
                "workflow_type": "pipeline",
                "completed_steps": completed_steps,
                "step_results": safe_pipeline_results,
            },
        )
        listener.info("Pipeline completed successfully")

        return PipelineOutput(
            success=True,
            step_results=safe_pipeline_results,
            completed_steps=completed_steps,
            skipped_steps=safe_skip_steps,
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
        # Load config - either from file or use provided dict
        if isinstance(params.config, dict):
            pipeline_config = params.config
        else:
            with open(params.config, encoding="utf-8") as f:
                pipeline_config = yaml.safe_load(f)

        # Detect config format (unified vs old)
        is_unified = "project" in pipeline_config

        steps_section = pipeline_config.get("steps") if is_unified else pipeline_config
        if steps_section is None:
            steps_section = {}
        elif not isinstance(steps_section, dict):
            raise ValueError("Pipeline configuration 'steps' must be a mapping")

        # Use centralized step names
        from material_agent.api.defaults import PIPELINE_STEP_NAMES

        step_names = PIPELINE_STEP_NAMES

        planned_steps = []
        skipped_steps = []

        for step in step_names:
            if step not in steps_section:
                continue

            step_config = steps_section[step]

            # Check if enabled (for unified format)
            if is_unified:
                enabled = step_config.get("enabled")
                if enabled is None:
                    # Implicitly enable if step has any configuration besides 'enabled'
                    has_config = any(k != "enabled" for k in step_config.keys())
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
    """Execute a multi-step material agent pipeline synchronously.

    This is a wrapper around the async implementation for backward compatibility.

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
    resume: bool = False,
    dry_run: bool = False,
    clean: bool = False,
    event_listener: EventListener | None = None,
    verbose: bool = False,
    cancel_checker: Callable[[], bool] | None = None,
) -> PipelineOutput:
    """Async convenience function for pipeline API.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output
        cancel_checker: Optional callback returning True when the run should stop

    Returns:
        PipelineOutput with results
    """
    params = PipelineInput(
        config=config,
        skip_steps=skip_steps or [],
        only_steps=only_steps or [],
        resume=resume,
        dry_run=dry_run,
        clean=clean,
        event_listener=event_listener,
        verbose=verbose,
        cancel_checker=cancel_checker,
    )
    return await arun_pipeline(params)


def pipeline(
    config: Path | dict[str, Any],
    skip_steps: list[str] | None = None,
    only_steps: list[str] | None = None,
    resume: bool = False,
    dry_run: bool = False,
    clean: bool = False,
    event_listener: EventListener | None = None,
    verbose: bool = False,
    cancel_checker: Callable[[], bool] | None = None,
) -> PipelineOutput:
    """Sync convenience function for pipeline API.

    This delegates to the async version for implementation reuse.

    Args:
        config: Either a Path to a YAML config file or a dict with config contents
        skip_steps: List of step names to skip
        only_steps: List of step names to run exclusively
        resume: Resume from last checkpoint
        dry_run: Show execution plan without running
        clean: Clean working directory before starting
        event_listener: Optional event listener for progress reporting
        verbose: Enable verbose output
        cancel_checker: Optional callback returning True when the run should stop

    Returns:
        PipelineOutput with results
    """
    return asyncio.run(
        apipeline(
            config,
            skip_steps,
            only_steps,
            resume,
            dry_run,
            clean,
            event_listener,
            verbose,
            cancel_checker,
        )
    )
