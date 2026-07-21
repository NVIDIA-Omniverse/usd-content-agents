# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base pipeline executor with common checkpoint/resume/clean logic.

This module provides a base class for pipeline executors that handles:
- State persistence (checkpoint/resume)
- Clean directory operations with safety checks
- Step filtering (skip/only)
- Context validation
- Common execution loop structure

Agent-specific executors inherit from this base and implement step execution logic.
"""

import fcntl
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

from filelock import FileLock, Timeout
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from world_understanding.agentic.tasks import Task
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    confined_cleanup_path,
    open_confined_directory,
    open_confined_lock_file,
    remove_confined_tree,
    remove_legacy_pipeline_temp,
    write_bytes_to_confined,
)
from world_understanding.utils.credentials import (
    InlineSecretError,
    ensure_no_inline_secrets,
    path_exists_with_safe_diagnostics,
    redact_sensitive_config,
    redact_sensitive_path,
    resolve_path_with_safe_diagnostics,
)
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    is_model_authentication_error,
)
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class _SafePipelineRuntimeError(RuntimeError):
    """Private marker for a code-owned runtime diagnostic."""


class _SafePipelineValueError(ValueError):
    """Private marker for a code-owned validation diagnostic."""


def _raise_runtime_error(message: str) -> NoReturn:
    """Raise a detached runtime error from a value-free helper frame."""
    raise _SafePipelineRuntimeError(message) from None


def _raise_value_error(message: str) -> NoReturn:
    """Raise a detached value error from a value-free helper frame."""
    raise _SafePipelineValueError(message) from None


def _raise_inline_secret_error(message: str) -> NoReturn:
    """Raise a detached credential rejection without retaining rejected data."""
    raise InlineSecretError(message) from None


def _raise_os_error(
    error_type: type[OSError],
    error_number: int | None,
    message: str,
    safe_path: str | None,
    safe_path2: str | None = None,
) -> NoReturn:
    """Raise a detached OS error containing only operation-owned diagnostics."""
    if safe_path2 is not None:
        raise error_type(error_number, message, safe_path, None, safe_path2) from None
    if safe_path is None:
        raise error_type(error_number, message) from None
    raise error_type(error_number, message, safe_path) from None


def _diagnostic_text(value: Any) -> str:
    """Project one runtime value to a credential-safe diagnostic string."""
    try:
        projected = redact_sensitive_config(value)
        return redact_sensitive_path(str(projected))
    except Exception:  # pragma: no cover - defensive diagnostic boundary
        return "<unavailable>"


def _diagnostic_steps(steps: list[str]) -> list[str]:
    """Project runtime step identifiers without changing dispatch values."""
    return [_diagnostic_text(step) for step in steps]


def safe_diagnostic_text(value: Any) -> str:
    """Expose the shared credential-safe scalar diagnostic projection."""
    return _diagnostic_text(value)


def safe_diagnostic_steps(steps: list[str]) -> list[str]:
    """Expose the shared projection for step diagnostics and event payloads."""
    return _diagnostic_steps(steps)


def remove_legacy_pipeline_temp_with_safe_diagnostics(
    working_dir: str | Path,
) -> bool:
    """Remove retained legacy config transport without exposing its path."""
    legacy_temp = Path(working_dir) / ".pipeline_temp"
    safe_legacy_temp = redact_sensitive_path(legacy_temp)
    os_failure: tuple[type[OSError], int | None] | None = None
    try:
        return remove_legacy_pipeline_temp(working_dir)
    except OSError as error:
        os_failure = (type(error), error.errno)

    assert os_failure is not None
    error_type, error_number = os_failure
    del working_dir, legacy_temp, os_failure
    _raise_os_error(
        error_type,
        error_number,
        "Unable to remove retained legacy pipeline configuration",
        safe_legacy_temp,
    )


def safe_exception_category(error: BaseException) -> str:
    """Return a bounded code-defined exception category without its value."""
    category = type(error).__name__
    if len(category) > 128 or not category.isidentifier():
        return "Exception"
    return category


def safe_step_failure_message(error: BaseException) -> str:
    """Return a value-free failure summary for logs, events, and checkpoints.

    Provider and validator exceptions routinely include request configuration,
    prompts, URLs, and credentials in their text and traceback locals.  Pipeline
    boundaries may retain or publish their diagnostics, so callers must project
    only the stable exception category and never stringify ``error`` itself.
    """
    if is_model_authentication_error(error):
        return MODEL_AUTHENTICATION_FAILURE_MESSAGE
    return f"{safe_exception_category(error)} during step execution"


def _safe_public_exception_message(error: BaseException) -> str:
    """Project an implementation failure to a detached public message."""
    try:
        message = str(error)
    except Exception:
        return f"{safe_exception_category(error)} during pipeline execution"
    return _diagnostic_text(message)


def _raise_public_pipeline_exception(
    error_type: type[Exception],
    message: str,
) -> NoReturn:
    """Reconstruct a public exception without its implementation traceback."""
    replacement: Exception | None = None
    try:
        replacement = error_type(message)
    except Exception:
        # The documented executor errors all accept one string argument. Keep an
        # unexpected custom constructor failure value-free without reconnecting
        # it to the rejected implementation graph.
        replacement = None

    if replacement is None:
        _raise_runtime_error(message)
    raise replacement from None


class _CredentialRedactingLogFilter(logging.Filter):
    """Redact credential-bearing values from scoped dependency diagnostics."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive logging boundary
            message = "External lock diagnostic unavailable"
        record.msg = redact_sensitive_path(message)
        record.args = ()
        return True


@contextmanager
def _redact_filelock_diagnostics() -> Iterator[None]:
    """Keep FileLock's full-path debug messages credential-safe."""
    filelock_logger = logging.getLogger("filelock")
    diagnostic_filter = _CredentialRedactingLogFilter()
    filelock_logger.addFilter(diagnostic_filter)
    try:
        yield
    finally:
        filelock_logger.removeFilter(diagnostic_filter)


@contextmanager
def _confined_checkpoint_lock(
    parent_descriptor: int,
    lock_name: str,
    *,
    timeout: float,
) -> Iterator[None]:
    """Take a bounded checkpoint lock beneath one held parent descriptor."""
    with open_confined_lock_file(
        parent_descriptor,
        lock_name,
    ) as lock_descriptor:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(
                    lock_descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Timeout("<redacted>") from None
                time.sleep(min(0.01, remaining))
        try:
            yield
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)


# Get a tracer for pipeline operations
_tracer = trace.get_tracer(__name__)


def is_valid_pipeline_checkpoint_structure(value: Any) -> bool:
    """Return whether a decoded checkpoint has the shared resume shape."""
    if (
        type(value) is not dict
        or type(value.get("completed_steps")) is not list
        or not all(type(step_name) is str for step_name in value["completed_steps"])
        or type(value.get("failed_steps")) is not list
        or not all(type(step_name) is str for step_name in value["failed_steps"])
        or type(value.get("step_outputs")) is not dict
        or not all(
            type(step_name) is str and type(step_output) is dict
            for step_name, step_output in value["step_outputs"].items()
        )
        or (
            value.get("current_step") is not None
            and type(value["current_step"]) is not str
        )
    ):
        return False

    step_errors = value.get("step_errors", {})
    return type(step_errors) is dict and all(
        type(step_name) is str and type(step_error) is str
        for step_name, step_error in step_errors.items()
    )


class PathEncoder(json.JSONEncoder):
    """JSON encoder that handles Path objects.

    This encoder converts Path objects to strings when serializing to JSON,
    making it easier to persist pipeline state with file paths.
    """

    def default(self, obj: Any) -> Any:
        """Convert Path objects to strings."""
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_pipeline_checkpoint(
    pipeline_state: dict[str, Any],
    state_file: Path,
) -> None:
    """Validate and atomically publish one locked pipeline checkpoint."""
    safe_state_file = redact_sensitive_path(state_file)
    serialized: bytes | None = None
    failure_kind: str | None = None
    os_failure: tuple[type[OSError], int | None] | None = None
    try:
        with open_confined_directory(
            state_file.parent,
            create=True,
        ) as parent_descriptor:
            with _confined_checkpoint_lock(
                parent_descriptor,
                state_file.with_suffix(".lock").name,
                timeout=30,
            ):
                ensure_no_inline_secrets(
                    pipeline_state,
                    context="pipeline checkpoint state",
                )
                serialized = json.dumps(
                    pipeline_state,
                    indent=2,
                    cls=PathEncoder,
                ).encode("utf-8")
                write_bytes_to_confined(
                    parent_descriptor,
                    state_file.name,
                    serialized,
                    file_mode=0o600,
                )
            logger.debug("Checkpoint saved to: %s", safe_state_file)
    except Timeout:
        logger.error(
            "Timeout acquiring lock for %s. Another process may be "
            "accessing this session.",
            safe_state_file,
        )
        failure_kind = "timeout"
    except InlineSecretError:
        failure_kind = "inline-secret"
    except ArtifactPathError:
        failure_kind = "unsafe-path"
    except (TypeError, ValueError):
        failure_kind = "serialization"
    except OSError as error:
        os_failure = (type(error), error.errno)

    if failure_kind is None and os_failure is None:
        return

    del pipeline_state, serialized, state_file
    if os_failure is not None:
        error_type, error_number = os_failure
        _raise_os_error(
            error_type,
            error_number,
            "Unable to write pipeline checkpoint",
            safe_state_file,
        )
    if failure_kind == "inline-secret":
        _raise_inline_secret_error(
            "Pipeline checkpoint state contains inline credentials"
        )
    if failure_kind == "timeout":
        _raise_runtime_error(
            f"Could not save checkpoint to {safe_state_file} - lock timeout. "
            "Ensure no other pipelines are using the same session_id."
        )
    _raise_runtime_error("Unable to publish a valid pipeline checkpoint")


class BasePipelineExecutor(Task):
    """Base class for pipeline executors with common functionality.

    This class provides reusable infrastructure for multi-step pipeline execution:
    - **Checkpoint/Resume**: Save progress after each step, resume from failures
    - **Clean Operations**: Safe directory cleanup with validation
    - **Step Filtering**: Support for skip_steps and only_steps
    - **State Tracking**: Track completed steps, failures, and outputs

    Subclasses must implement:
    - `_execute_step()`: Execute a single pipeline step
    - `_get_step_list_key()`: Return context key for step list
    - `_get_required_context_keys()`: Return required context keys
    - `_get_state_file()`: Return path to state file

    Example:
        >>> class MyExecutor(BasePipelineExecutor):
        ...     def _execute_step(self, step_name, context, object_store):
        ...         # Execute step-specific logic
        ...         return {"status": "completed"}
        ...
        ...     def _get_step_list_key(self):
        ...         return "steps_to_run"
        ...
        ...     def _get_required_context_keys(self):
        ...         return ["steps_to_run", "config"]
        ...
        ...     def _get_state_file(self, context):
        ...         return context["working_dir"] / ".pipeline_state.json"
    """

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Execute pipeline steps without exporting implementation tracebacks.

        Runtime contexts and step results intentionally retain their exact values
        while a pipeline is running. If execution fails, the implementation
        traceback can therefore retain credentials even when the exception text
        is already safe. Reconstruct the public exception only after discarding
        that graph and every caller-owned runtime reference from this frame.

        Args:
            context: Workflow context with configuration
            object_store: Optional object store for workflow execution

        Returns:
            The same runtime context, updated with pipeline results

        Raises:
            ValueError: If required context keys are missing or validation fails
            RuntimeError: If a pipeline step fails
        """
        failure_type: type[Exception] | None = None
        failure_message: str | None = None
        os_failure: (
            tuple[
                type[OSError],
                int | None,
                str,
                str | None,
                str | None,
            ]
            | None
        ) = None
        try:
            return self._run_impl(context, object_store)
        except OSError as error:
            filename2 = getattr(error, "filename2", None)
            if (
                error.errno is None
                and error.strerror is None
                and error.filename is None
                and filename2 is None
            ):
                failure_type = type(error)
                failure_message = (
                    f"{safe_exception_category(error)} during pipeline operation"
                )
            else:
                safe_os_message = (
                    _diagnostic_text(error.strerror)
                    if error.strerror is not None
                    else "Pipeline operation failed"
                )
                safe_filename = (
                    None
                    if error.filename is None
                    else redact_sensitive_path(error.filename)
                )
                safe_filename2 = (
                    None if filename2 is None else redact_sensitive_path(filename2)
                )
                os_failure = (
                    type(error),
                    error.errno,
                    safe_os_message,
                    safe_filename,
                    safe_filename2,
                )
        except InlineSecretError as error:
            failure_type = type(error)
            failure_message = _safe_public_exception_message(error)
        except _SafePipelineValueError as error:
            failure_type = ValueError
            failure_message = _safe_public_exception_message(error)
        except _SafePipelineRuntimeError as error:
            failure_type = RuntimeError
            failure_message = _safe_public_exception_message(error)
        except Exception as error:
            failure_type = RuntimeError
            failure_message = (
                f"{safe_exception_category(error)} during pipeline execution"
            )

        del context, object_store, self
        if os_failure is not None:
            error_type, error_number, message, safe_path, safe_path2 = os_failure
            _raise_os_error(
                error_type,
                error_number,
                message,
                safe_path,
                safe_path2,
            )
        assert failure_type is not None
        assert failure_message is not None
        _raise_public_pipeline_exception(failure_type, failure_message)

    def _run_impl(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Execute pipeline steps in sequence.

        This method provides the common execution loop:
        1. Validate required context keys
        2. Apply step filtering (skip/only)
        3. Clean directories (if requested)
        4. Initialize or load pipeline state
        5. Execute each step in sequence
        6. Save checkpoint after each step
        7. Update context with results

        Args:
            context: Workflow context with configuration
            object_store: Optional object store for workflow execution

        Returns:
            Updated context with pipeline results

        Raises:
            ValueError: If required context keys are missing or validation fails
            RuntimeError: If a pipeline step fails
        """
        ensure_no_inline_secrets(
            context.get("session_id"),
            context="pipeline session identifier",
            path_context=True,
        )
        # Get pipeline name for tracing
        pipeline_name = _diagnostic_text(
            context.get("project_name", self.__class__.__name__)
        )

        # Start the main pipeline span
        with _tracer.start_as_current_span("pipeline.run") as pipeline_span:
            # Set pipeline-level attributes
            pipeline_span.set_attribute("maa.pipeline.name", pipeline_name)
            pipeline_span.set_attribute(
                "maa.pipeline.session_id",
                _diagnostic_text(context.get("session_id", "unknown")),
            )

            # 1. Validate required context keys
            self._validate_context(context)

            # 2. Get step list and apply filtering
            step_list_key = self._get_step_list_key()
            steps = context.get(step_list_key, [])
            if not steps:
                _raise_value_error(
                    f"No steps to run in pipeline ({step_list_key} is empty)"
                )

            steps = self._apply_step_filtering(steps, context)
            logger.info(
                "Pipeline will execute %s steps: %s",
                len(steps),
                _diagnostic_steps(steps),
            )

            # Set total steps attribute after filtering
            pipeline_span.set_attribute("maa.pipeline.total_steps", len(steps))
            pipeline_span.set_attribute(
                "maa.pipeline.steps",
                ",".join(_diagnostic_steps(steps)),
            )

            # 3. Clean directories if requested
            if context.get("clean", False):
                self._clean_directories(context)

            # 4. Initialize or load pipeline state
            resume = context.get("resume", False)
            pipeline_state = self._initialize_pipeline_state(context, resume)

            # 5. Execute each step
            state_file = self._get_state_file(context)
            self._log_pipeline_started(context, steps)

            for i, step_name in enumerate(steps, 1):
                # Skip if resuming and step already completed
                if resume and step_name in pipeline_state["completed_steps"]:
                    logger.info(
                        "[%s/%s] Skipping completed step: %s",
                        i,
                        len(steps),
                        _diagnostic_text(step_name),
                    )
                    continue

                # Execute step with tracing
                logger.info(
                    "\n[%s/%s] Executing step: %s",
                    i,
                    len(steps),
                    _diagnostic_text(step_name),
                )
                pipeline_state["current_step"] = step_name

                self._execute_step_with_tracing(
                    step_name=step_name,
                    step_index=i - 1,  # 0-based index for attributes
                    total_steps=len(steps),
                    context=context,
                    object_store=object_store,
                    pipeline_state=pipeline_state,
                    state_file=state_file,
                )

            # 6. Mark pipeline as completed
            pipeline_state["current_step"] = None
            self._save_checkpoint(pipeline_state, state_file)

            # 7. Update context with results
            self._log_pipeline_completed(context, pipeline_state)
            self._update_context_with_results(context, pipeline_state)

            # Set final pipeline status
            completed_count = len(pipeline_state["completed_steps"])
            failed_count = len(pipeline_state["failed_steps"])
            pipeline_span.set_attribute("maa.pipeline.completed_steps", completed_count)
            pipeline_span.set_attribute("maa.pipeline.failed_steps", failed_count)
            pipeline_span.set_attribute("maa.pipeline.status", "completed")

            return context

    def _execute_step_with_tracing(
        self,
        step_name: str,
        step_index: int,
        total_steps: int,
        context: dict[str, Any],
        object_store: ObjectStore | None,
        pipeline_state: dict[str, Any],
        state_file: Path,
    ) -> dict[str, Any]:
        """Execute a single pipeline step with OpenTelemetry tracing.

        Args:
            step_name: Name of the step to execute
            step_index: 0-based index of the step in the pipeline
            total_steps: Total number of steps in the pipeline
            context: Workflow context
            object_store: Optional object store
            pipeline_state: Current pipeline state dictionary
            state_file: Path to the state file for checkpointing

        Returns:
            Dictionary with step results/outputs

        Raises:
            RuntimeError: If the step fails
        """
        safe_step_name = _diagnostic_text(step_name)
        with _tracer.start_as_current_span(
            f"pipeline.step.{safe_step_name}"
        ) as step_span:
            # Set step-level attributes
            step_span.set_attribute("maa.pipeline.step.name", safe_step_name)
            step_span.set_attribute("maa.pipeline.step.index", step_index)
            step_span.set_attribute("maa.pipeline.step.total", total_steps)

            safe_error: str | None = None
            step_result: dict[str, Any] | None = None
            try:
                step_result = self._execute_step(step_name, context, object_store)
            except Exception as error:
                # Arbitrary exception text is an untrusted diagnostic surface:
                # providers and validators commonly embed input values in it.
                # Retain only the category, then leave the rejected exception
                # handler before performing failure bookkeeping or publishing a
                # replacement exception. ``raise ... from None`` inside this
                # handler would still retain ``error`` through ``__context__``.
                safe_error = safe_step_failure_message(error)
            else:
                # Checkpoint persistence is not step execution. In particular, a
                # credential rejection must propagate once to the public run
                # boundary instead of being misclassified as a step failure and
                # retried with the same rejected state.
                # Track completion
                pipeline_state["completed_steps"].append(step_name)
                pipeline_state["step_outputs"][step_name] = step_result

                # Save checkpoint
                self._save_checkpoint(pipeline_state, state_file)

                logger.info("Step '%s' completed successfully", safe_step_name)

                # Set success attributes
                step_span.set_attribute("maa.pipeline.step.status", "completed")

                return step_result

            assert safe_error is not None

            # Track and publish failure only after the rejected exception has
            # left the active handler, severing it from every replacement
            # diagnostic and exception object.
            pipeline_state["failed_steps"].append(step_name)
            self._save_checkpoint(pipeline_state, state_file)
            logger.error("Step '%s' failed: %s", safe_step_name, safe_error)

            # Set failure attributes and record exception
            step_span.set_attribute("maa.pipeline.step.status", "failed")
            step_span.record_exception(RuntimeError(safe_error))
            step_span.set_status(Status(StatusCode.ERROR, safe_error))

            failure_message = (
                f"Pipeline failed at step '{safe_step_name}': {safe_error}"
            )
            # The replacement traceback includes this frame. Remove every raw
            # runtime reference before raising so exception collectors cannot
            # recover credentials from frame locals even though the rejected
            # exception graph itself has already been severed.
            del (
                context,
                object_store,
                pipeline_state,
                self,
                state_file,
                step_name,
                step_result,
            )
            _raise_runtime_error(failure_message)

    # ========== Abstract Methods (must implement in subclass) ==========

    def _execute_step(
        self,
        step_name: str,
        context: dict[str, Any],
        object_store: ObjectStore | None,
    ) -> dict[str, Any]:
        """Execute a single pipeline step (agent-specific logic).

        Args:
            step_name: Name of the step to execute
            context: Workflow context
            object_store: Optional object store

        Returns:
            Dictionary with step results/outputs

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _execute_step()"
        )

    def _get_step_list_key(self) -> str:
        """Return context key for step list.

        Returns:
            Context key name (e.g., 'steps_to_run' or 'enabled_steps')

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_step_list_key()"
        )

    def _get_required_context_keys(self) -> list[str]:
        """Return list of required context keys for validation.

        Returns:
            List of required context key names

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_required_context_keys()"
        )

    def _get_state_file(self, context: dict[str, Any]) -> Path:
        """Return path to pipeline state file.

        Args:
            context: Workflow context

        Returns:
            Path to state file (typically .pipeline_state.json)

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_state_file()"
        )

    # ========== Optional Override Methods (have default implementations) ==========

    def _update_context_with_results(
        self, context: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> None:
        """Update context with pipeline results.

        Default implementation stores results in 'pipeline_results' key.
        Subclasses can override to customize result storage.

        Args:
            context: Workflow context to update
            pipeline_state: Final pipeline state
        """
        context["pipeline_results"] = pipeline_state["step_outputs"]
        context["pipeline_state"] = "completed"

    # ========== Common Methods (implemented in base class) ==========

    def _get_state_lock_file(self, state_file: Path) -> Path:
        """Get path to lock file for state file.

        Args:
            state_file: Path to state file

        Returns:
            Path to corresponding lock file
        """
        return state_file.with_suffix(".lock")

    def _validate_context(self, context: dict[str, Any]) -> None:
        """Validate that required context keys are present.

        Args:
            context: Workflow context to validate

        Raises:
            ValueError: If required context keys are missing
        """
        required_keys = self._get_required_context_keys()
        missing_keys = [key for key in required_keys if key not in context]

        if missing_keys:
            _raise_value_error(
                f"Required context keys missing: {missing_keys}. "
                f"Required: {required_keys}"
            )

    def _apply_step_filtering(
        self, steps: list[str], context: dict[str, Any]
    ) -> list[str]:
        """Apply skip_steps and only_steps filtering to step list.

        Args:
            steps: Original list of steps
            context: Workflow context with optional skip_steps/only_steps

        Returns:
            Filtered list of steps
        """
        skip_steps = context.get("skip_steps", [])
        only_steps = context.get("only_steps", [])

        # Apply only_steps filter first (if provided)
        if only_steps:
            steps = [step for step in steps if step in only_steps]
            logger.debug("Filtered to only steps: %s", _diagnostic_steps(steps))

        # Apply skip_steps filter
        if skip_steps:
            steps = [step for step in steps if step not in skip_steps]
            logger.debug("Skipped steps: %s", _diagnostic_steps(skip_steps))

        return steps

    def _clean_directories(self, context: dict[str, Any]) -> None:
        """Clean working directory and output files with safety checks.

        This performs a clean operation to remove previous pipeline outputs.
        Includes safety checks to prevent accidental deletion of important paths.

        Args:
            context: Workflow context with working_dir

        Raises:
            ValueError: If working_dir path is unsafe to delete
        """
        working_dir = context.get("working_dir")
        if not working_dir:
            logger.warning("No working_dir in context, skipping clean operation")
            return

        working_dir_path = Path(working_dir)
        safe_working_dir = redact_sensitive_path(working_dir_path)
        canonical_working_dir = resolve_path_with_safe_diagnostics(
            working_dir_path,
            label="pipeline working directory",
        )
        path_resolver = context.get("path_resolver")
        cleanup_root = (
            context.get("working_dir_base")
            or getattr(path_resolver, "working_dir_base", None)
            or getattr(path_resolver, "config_dir", None)
        )

        # Safety decisions must use the canonical target. A lexical alias such
        # as ``/tmp/..`` or a symlink to the home directory is just as
        # dangerous as spelling the target directly. Reject critical targets
        # before considering the supplied ownership root so propagation cannot
        # accidentally authorize deletion of the filesystem root or home.
        if canonical_working_dir in {
            Path.home().resolve(),
            Path("/"),
        }:
            failure_message = (
                f"Refusing to delete potentially dangerous path: {safe_working_dir}"
            )
            del (
                canonical_working_dir,
                cleanup_root,
                context,
                path_resolver,
                self,
                working_dir,
                working_dir_path,
            )
            _raise_value_error(failure_message)

        # Preserve the caller-facing shallow-path guard before canonicalization:
        # a relative single-component path such as ``work`` resolves deeply
        # beneath the checkout but is still too broad an input for cleanup.
        if cleanup_root is None and (
            len(working_dir_path.parts) < 2 or len(canonical_working_dir.parts) < 2
        ):
            failure_message = f"Working directory path too shallow: {safe_working_dir}"
            del (
                canonical_working_dir,
                cleanup_root,
                context,
                path_resolver,
                self,
                working_dir,
                working_dir_path,
            )
            _raise_value_error(failure_message)

        if cleanup_root is None:
            del (
                canonical_working_dir,
                cleanup_root,
                context,
                path_resolver,
                self,
                working_dir,
                working_dir_path,
            )
            _raise_value_error(
                "Refusing recursive cleanup without a configured ownership root"
            )

        cleanup_root_path = resolve_path_with_safe_diagnostics(
            cleanup_root,
            label="pipeline cleanup ownership root",
        )
        validation_failure: str | None = None
        try:
            working_dir_path = confined_cleanup_path(
                working_dir_path,
                cleanup_root_path,
            )
        except ValueError:
            if canonical_working_dir == cleanup_root_path:
                validation_failure = (
                    "Working directory must be a child of the cleanup root"
                )
            else:
                validation_failure = (
                    "Working directory is outside the configured cleanup root"
                )

        if validation_failure is not None:
            del (
                canonical_working_dir,
                cleanup_root,
                cleanup_root_path,
                context,
                path_resolver,
                self,
                working_dir,
                working_dir_path,
            )
            _raise_value_error(validation_failure)

        # Clean working directory
        os_failure: tuple[type[OSError], int | None] | None = None
        try:
            logger.info("Cleaning working directory: %s", safe_working_dir)
            if remove_confined_tree(working_dir_path, cleanup_root_path):
                logger.info("Working directory cleaned successfully")
        except OSError as error:
            os_failure = (type(error), error.errno)
        except ArtifactPathError:
            # A validated ancestor changed before descriptor traversal. Treat
            # this deterministic race as a cleanup I/O failure while keeping
            # the rejected path graph off the public traceback.
            os_failure = (OSError, None)

        if os_failure is not None:
            error_type, error_number = os_failure
            del (
                canonical_working_dir,
                cleanup_root,
                cleanup_root_path,
                context,
                os_failure,
                path_resolver,
                self,
                working_dir,
                working_dir_path,
            )
            _raise_os_error(
                error_type,
                error_number,
                "Unable to clean working directory",
                safe_working_dir,
            )

    def _initialize_pipeline_state(
        self, context: dict[str, Any], resume: bool = False
    ) -> dict[str, Any]:
        """Initialize or load pipeline state with file locking.

        Args:
            context: Workflow context
            resume: If True, attempt to load existing state

        Returns:
            Pipeline state dictionary

        Raises:
            RuntimeError: If unable to acquire lock within timeout period
        """
        state_file = self._get_state_file(context)
        safe_state_file = redact_sensitive_path(state_file)

        # Try to load existing state if resuming
        state_exists = resume and path_exists_with_safe_diagnostics(
            state_file,
            label="pipeline checkpoint",
        )

        if state_exists:
            lock_file = self._get_state_lock_file(state_file)
            checkpoint_file: Any = None
            loaded_value: Any = None
            loaded_state: dict[str, Any] | None = None
            failure_kind: str | None = None
            os_failure: tuple[type[OSError], int | None] | None = None
            try:
                # Timeout ensures we never block indefinitely
                with (
                    _redact_filelock_diagnostics(),
                    FileLock(str(lock_file), timeout=30),
                ):
                    logger.info(
                        "Resuming pipeline from checkpoint: %s", safe_state_file
                    )
                    with open(state_file, encoding="utf-8") as checkpoint_file:
                        loaded_value = json.load(checkpoint_file)

                    if type(loaded_value) is not dict:
                        failure_kind = "root"
                    elif not is_valid_pipeline_checkpoint_structure(loaded_value):
                        failure_kind = "structure"
                    else:
                        loaded_state = loaded_value
                        try:
                            ensure_no_inline_secrets(
                                loaded_state,
                                context="pipeline resume state",
                            )
                        except InlineSecretError:
                            failure_kind = "inline-secret"

                        if failure_kind is None:
                            logger.info(
                                "Resumed with %d completed steps",
                                len(loaded_state.get("completed_steps", [])),
                            )
                            return loaded_state

            except Timeout:
                # After 30 seconds, give up gracefully
                logger.error(
                    "Timeout acquiring lock for %s. Another process may be "
                    "reading/writing the file.",
                    safe_state_file,
                )
                failure_kind = "timeout"
            except json.JSONDecodeError:
                failure_kind = "json"
            except OSError as error:
                os_failure = (type(error), error.errno)

            if failure_kind is not None or os_failure is not None:
                # This method is itself retained on the replacement traceback.
                # Drop both the rejected checkpoint and every runtime-derived
                # path/context reference before delegating the public raise to
                # a value-free helper frame.
                del (
                    checkpoint_file,
                    context,
                    loaded_state,
                    loaded_value,
                    lock_file,
                    self,
                    state_file,
                )
                if os_failure is not None:
                    error_type, error_number = os_failure
                    _raise_os_error(
                        error_type,
                        error_number,
                        "Unable to read pipeline checkpoint",
                        safe_state_file,
                    )
                if failure_kind == "timeout":
                    _raise_runtime_error(
                        "Failed to resume - could not acquire lock on "
                        f"{safe_state_file}"
                    )
                if failure_kind == "json":
                    _raise_value_error(
                        f"Unable to parse pipeline checkpoint: {safe_state_file}"
                    )
                if failure_kind == "inline-secret":
                    _raise_inline_secret_error(
                        "Pipeline resume state contains inline credentials"
                    )
                if failure_kind == "root":
                    _raise_value_error("Pipeline checkpoint root must be a JSON object")
                _raise_value_error(
                    f"Invalid pipeline checkpoint structure: {safe_state_file}"
                )
        # Initialize new state
        return {
            "session_id": context.get("session_id", "unknown"),
            "project_name": context.get("project_name", "unknown"),
            "completed_steps": [],
            "failed_steps": [],
            "step_outputs": {},
            "current_step": None,
        }

    def _save_checkpoint(
        self, pipeline_state: dict[str, Any], state_file: Path
    ) -> None:
        """Save pipeline state checkpoint to disk with file locking.

        Args:
            pipeline_state: Current pipeline state
            state_file: Path to state file

        Raises:
            RuntimeError: If unable to acquire lock within timeout period
        """
        try:
            save_pipeline_checkpoint(pipeline_state, state_file)
        except BaseException:
            del pipeline_state, self, state_file
            raise

    def _log_pipeline_started(self, context: dict[str, Any], steps: list[str]) -> None:
        """Log pipeline start banner.

        Args:
            context: Workflow context
            steps: List of steps to execute
        """
        logger.info("\n" + "=" * 80)
        project_name = _diagnostic_text(context.get("project_name", "unknown"))
        session_id = _diagnostic_text(context.get("session_id", "unknown"))
        logger.info("Starting pipeline: %s (session: %s)", project_name, session_id)
        logger.info("Steps to execute: %s", ", ".join(_diagnostic_steps(steps)))
        logger.info("=" * 80 + "\n")

    def _log_pipeline_completed(
        self, context: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> None:
        """Log pipeline completion banner.

        Args:
            context: Workflow context
            pipeline_state: Final pipeline state
        """
        completed = len(pipeline_state["completed_steps"])
        failed = len(pipeline_state["failed_steps"])

        logger.info("\n" + "=" * 80)
        project_name = _diagnostic_text(context.get("project_name", "unknown"))
        session_id = _diagnostic_text(context.get("session_id", "unknown"))
        logger.info("Pipeline completed: %s (session: %s)", project_name, session_id)
        logger.info(f"Completed steps: {completed}")
        if failed > 0:
            logger.warning(f"Failed steps: {failed}")
        logger.info("=" * 80 + "\n")
