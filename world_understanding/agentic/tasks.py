# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task implementations for workflow execution."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from world_understanding.agentic.base import BaseAgent
from world_understanding.tools.base import get_tool_registry
from world_understanding.utils.credentials import redact_sensitive_path
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)

_TOOL_EXECUTION_FAILURE_MESSAGE = "Tool execution failed"


def _diagnostic_name(value: Any) -> str:
    """Project a runtime tool identifier onto observable surfaces."""
    if not isinstance(value, str):
        return "<unavailable>"
    return redact_sensitive_path(value)


class Task(ABC):
    """Abstract base class for workflow tasks."""

    @abstractmethod
    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the task synchronously.

        Args:
            context: Workflow context to read from and update
            object_store: Storage for artifacts

        Returns:
            Updated context
        """
        pass

    async def arun(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the task asynchronously.

        Default implementation delegates to sync run() via asyncio.to_thread.
        Subclasses can override for true async behavior.

        Args:
            context: Workflow context to read from and update
            object_store: Storage for artifacts

        Returns:
            Updated context
        """
        return await asyncio.to_thread(self.run, context, object_store)


class AgenticLoopTask(Task):
    """
    Task that runs an agent in a loop with confidence-based refinement.

    This is an application-specific task that demonstrates how to use agents
    for iterative refinement based on confidence scores.
    """

    def __init__(
        self,
        planner_agent: BaseAgent,
        max_iterations: int = 5,
        early_exit_threshold: float = 0.85,
    ):
        """
        Initialize the agentic loop task.

        Args:
            planner_agent: Agent to execute the workflow
            max_iterations: Maximum refinement iterations
            early_exit_threshold: Confidence threshold for early exit
        """
        self.planner_agent = planner_agent
        self.max_iterations = max_iterations
        self.early_exit_threshold = early_exit_threshold

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the agent in a loop with refinement synchronously.

        This is a wrapper around the async implementation.

        The agent is expected to set:
        - context["avg_confidence"]: Average confidence score
        - context["completed"]: True if workflow is complete
        - context["needs_refinement"]: True if refinement needed
        """
        return asyncio.run(self.arun(context, object_store))

    async def arun(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the agent in a loop with refinement asynchronously.

        The agent is expected to set:
        - context["avg_confidence"]: Average confidence score
        - context["completed"]: True if workflow is complete
        - context["needs_refinement"]: True if refinement needed
        """
        if object_store is None:
            from world_understanding.utils.object_store import InMemoryObjectStore

            object_store = InMemoryObjectStore()

        for iteration in range(self.max_iterations):
            context["refinement_iteration"] = iteration

            # Execute agent asynchronously
            context = await self.planner_agent.arun(
                task="execute_workflow", context=context, object_store=object_store
            )

            # Check for early exit based on confidence
            avg_confidence = context.get("avg_confidence", 0.0)
            if avg_confidence >= self.early_exit_threshold:
                context["completed"] = True
                context["early_exit"] = True
                context["completion_reason"] = "confidence_threshold_met"
                break

            # Check if agent marked as completed
            if context.get("completed", False):
                break

            # Check if refinement is needed
            if not context.get("needs_refinement", False):
                # Agent didn't request refinement but also didn't complete
                context["completed"] = True
                context["completion_reason"] = "no_refinement_requested"
                break
        else:
            # Max iterations reached
            context["completed"] = True
            context["completion_reason"] = "max_iterations_reached"
            context["final_iteration"] = self.max_iterations

        return context


class CallableTask(Task):
    """Task that executes a callable function."""

    def __init__(self, func: callable, name: str = "CallableTask"):
        """
        Initialize a callable task.

        Args:
            func: Function to execute (should accept context and object_store)
            name: Task name
        """
        self.func = func
        self.name = name

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Execute the function."""
        return self.func(context, object_store)


class ToolTask(Task):
    """
    Task that executes a tool with support for dynamic input resolution.

    This enhanced ToolTask allows chaining tools together by referencing
    outputs from previous tasks in the workflow. It supports both static
    inputs and dynamic resolution from the workflow context.

    Features:
    - Static values provided at initialization
    - References to context values using "${context_key}" syntax
    - Nested references using "${key.subkey.subsubkey}" syntax
    - Custom output keys for organizing results
    - Input mapping for complex parameter resolution

    Examples:
        # Basic usage with static inputs
        extract_task = ToolTask(
            tool_name="extract_document_content",
            inputs={
                "source": "/path/to/docs",
                "output_dir": "/path/to/output",
                "save_content_only": True,
            },
            output_key="extraction_result",
        )

        # Using references to previous task outputs
        process_task = ToolTask(
            tool_name="process_data",
            inputs={
                # Reference to previous task's output
                "input_file": "${extraction_result.output_file}",
                "format": "json",  # Static value
            },
            output_key="processed_data",
        )

        # Using input_mapping for complex resolution
        final_task = ToolTask(
            tool_name="build_index",
            inputs={"base_dir": "/output"},
            input_mapping={
                # Map tool parameters to context values
                "data_file": "processed_data.file_path",
                "metadata": "${extraction_result.metadata}",
            },
            output_key="index_result",
        )

        # Chain tasks in a workflow
        workflow = Workflow(
            tasks=[extract_task, process_task, final_task],
            name="Document Processing Pipeline",
        )
        result = workflow.run()

        # Access results using custom keys
        extraction = result["extraction_result"]
        processed = result["processed_data"]
        index = result["index_result"]
    """

    def __init__(
        self,
        tool_name: str,
        inputs: dict[str, Any] | None = None,
        input_mapping: dict[str, str] | None = None,
        output_key: str | None = None,
        name: str | None = None,
    ):
        """
        Initialize a tool task with dynamic input resolution.

        Args:
            tool_name: Name of the tool to execute (must be registered)
            inputs: Static input parameters for the tool. Can contain:
                   - Direct values: {"param": "value"}
                   - References: {"param": "${context_key}"}
                   - Nested refs: {"param": "${result.data.field}"}
            input_mapping: Additional mapping from tool params to context keys.
                          Used to map tool parameters to values from previous tasks:
                          {"tool_param": "context_key"} or
                          {"tool_param": "${previous_result.field}"}
            output_key: Key to store output in context (defaults to "{tool_name}_result").
                       This key can be referenced by subsequent tasks.
            name: Optional task name for logging (defaults to tool_name)

        Example:
            # Task that uses output from a previous "extraction" task
            ToolTask(
                tool_name="process_content",
                inputs={
                    "static_param": "fixed_value",
                    "dynamic_param": "${extraction_result.file_path}",
                },
                input_mapping={
                    "metadata": "extraction_result.metadata",
                },
                output_key="processing_result",
                name="Process Extracted Content",
            )
        """
        self.tool_name = tool_name
        self.static_inputs = inputs or {}
        self.input_mapping = input_mapping or {}
        self.output_key = output_key or f"{tool_name}_result"
        self.name = name or tool_name
        self.tools = get_tool_registry()

    def _resolve_value(self, value: Any, context: dict[str, Any]) -> Any:
        """
        Resolve a value that might be a reference to context.

        This method handles the ${} reference syntax for dynamic value resolution.
        Non-string values and strings without ${} are returned unchanged.

        Supports:
        - Direct values: returned as-is (e.g., 42, True, ["list"])
        - Simple references: "${key}" resolves to context[key]
        - Nested references: "${key.subkey.field}" navigates through dicts

        Examples:
            context = {
                "task1_result": {
                    "output_file": "/path/to/file.json",
                    "metadata": {"count": 10, "status": "success"}
                }
            }

            _resolve_value("static_string", context) -> "static_string"
            _resolve_value(42, context) -> 42
            _resolve_value("${task1_result}", context) -> {"output_file": ..., "metadata": ...}
            _resolve_value("${task1_result.output_file}", context) -> "/path/to/file.json"
            _resolve_value("${task1_result.metadata.count}", context) -> 10

        Args:
            value: The value to resolve (can be any type)
            context: The workflow context containing previous task results

        Returns:
            The resolved value, or None if the reference couldn't be resolved
        """
        if not isinstance(value, str):
            return value

        if not (value.startswith("${") and value.endswith("}")):
            return value

        # Extract reference path
        # Extract reference path
        ref_path = value[2:-1].strip()  # Remove ${ and }

        # Validate reference format
        if (
            not ref_path
            or ref_path.startswith(".")
            or ref_path.endswith(".")
            or ".." in ref_path
        ):
            logger.warning("Invalid tool input reference format")
            return None

        # Split by dots for nested access
        keys = ref_path.split(".")

        # Navigate through context
        current = context
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                logger.warning("Could not resolve tool input reference")
                return None

        return current

    def _resolve_inputs(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve all inputs from static inputs and context mapping.

        This method combines static inputs with dynamically resolved values
        from the context. It processes both the 'inputs' and 'input_mapping'
        parameters to create the final set of parameters for the tool.

        Resolution order:
        1. Static inputs are processed first (may contain ${} references)
        2. Input mappings are applied second (can override static inputs)

        Examples:
            # Given this task configuration:
            task = ToolTask(
                tool_name="process_data",
                inputs={
                    "format": "json",  # Static value
                    "input_file": "${extraction.output_file}",  # Reference
                },
                input_mapping={
                    "metadata": "extraction.metadata",  # Direct mapping
                    "config": "${settings.process_config}",  # Reference mapping
                }
            )

            # With context:
            context = {
                "extraction": {
                    "output_file": "/data/extracted.json",
                    "metadata": {"pages": 10}
                },
                "settings": {
                    "process_config": {"quality": "high"}
                }
            }

            # Results in:
            {
                "format": "json",
                "input_file": "/data/extracted.json",
                "metadata": {"pages": 10},
                "config": {"quality": "high"}
            }

        Args:
            context: The workflow context containing all task results

        Returns:
            Dictionary of resolved input parameters for the tool
        """
        resolved = {}

        # Start with static inputs (which may contain references)
        for key, value in self.static_inputs.items():
            resolved[key] = self._resolve_value(value, context)

        # Apply input mapping from context
        for tool_param, context_ref in self.input_mapping.items():
            if isinstance(context_ref, str):
                # Could be a reference or a direct context key
                if context_ref.startswith("${"):
                    resolved[tool_param] = self._resolve_value(context_ref, context)
                else:
                    # Direct context key (for backward compatibility)
                    if "." in context_ref:
                        # Handle nested keys
                        keys = context_ref.split(".")
                        current = context
                        for k in keys:
                            if isinstance(current, dict) and k in current:
                                current = current[k]
                            else:
                                current = None
                                break
                        resolved[tool_param] = current
                    else:
                        resolved[tool_param] = context.get(context_ref)
            else:
                # Direct value
                resolved[tool_param] = context_ref

        return resolved

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the tool with dynamically resolved inputs synchronously.

        This is a wrapper around the async implementation for backward
        compatibility.

        Args:
            context: Workflow context containing all previous task results
            object_store: Optional storage for large artifacts

        Returns:
            Updated context with this task's results added under output_key
        """
        return asyncio.run(self.arun(context, object_store))

    async def arun(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the tool with dynamically resolved inputs asynchronously.

        This method is called by the Workflow during task execution. It:
        1. Resolves all inputs using context from previous tasks
        2. Executes the tool with resolved parameters
        3. Stores the result in context using the output_key
        4. Updates success/error flags for workflow control

        The context accumulates results from all previous tasks, allowing
        later tasks to reference earlier outputs. Each task's result is
        stored under its output_key (or "{tool_name}_result" by default).

        Context structure example after multiple tasks:
            {
                "workflow_name": "Document Pipeline",
                "current_task": "Build Index",
                "task_index": 2,

                # Results from previous tasks (using custom output_keys)
                "extraction_result": {
                    "document_count": 10,
                    "content_types": {"text": 8, "image": 2},
                    "output_file": "/output/extracted.json"
                },
                "processing_result": {
                    "processed_count": 10,
                    "output_dir": "/output/processed/"
                },

                # Success flags for each tool
                "extract_document_content_success": True,
                "process_data_success": True,

                # Current task will add its results here
            }

        Args:
            context: Workflow context containing all previous task results
            object_store: Optional storage for large artifacts

        Returns:
            Updated context with this task's results added under output_key
        """
        safe_tool_name = _diagnostic_name(self.tool_name)

        # Get the tool
        tool = self.tools.get(self.tool_name)
        if not tool:
            logger.error("Tool '%s' not found in registry", safe_tool_name)
            context["error"] = f"Tool '{safe_tool_name}' not found"
            context["failed_task"] = self.name
            return context

        try:
            # Resolve inputs from context
            resolved_inputs = self._resolve_inputs(context)

            logger.info("Executing tool: %s", safe_tool_name)
            # Resolved values are runtime data-plane inputs and may contain
            # credentials. Keep them raw for tool execution, but never render
            # them on a diagnostic surface.
            logger.debug("Resolved %d tool input field(s)", len(resolved_inputs))

            # Create input object for the tool
            input_obj = tool.spec.input_model(**resolved_inputs)

            # Execute the tool asynchronously
            output = await tool.arun(input_obj)

            # Convert output to dict if it's a Pydantic model
            if hasattr(output, "model_dump"):
                output_dict = output.model_dump()
            else:
                output_dict = output

            # Store results in context with custom key
            context[self.output_key] = output_dict
            context[f"{self.tool_name}_success"] = True

            # Store in object store if provided
            if object_store:
                object_store.set(self.output_key, output_dict)
                logger.debug("Stored tool output in object store")

            logger.info("Tool %s executed successfully", safe_tool_name)

        except Exception:
            # Validation, tool, and object-store exceptions can embed resolved
            # configuration values. Preserve the existing non-raising ToolTask
            # contract while returning only operation-owned diagnostics.
            logger.error("Tool %s failed", safe_tool_name)
            context[f"{self.tool_name}_error"] = _TOOL_EXECUTION_FAILURE_MESSAGE
            context[f"{self.tool_name}_success"] = False
            context["error"] = _TOOL_EXECUTION_FAILURE_MESSAGE
            context["failed_task"] = self.name

        return context


class RouterTask(Task):
    """
    Task that uses a RouterAgent to process multiple subtasks.

    This task enables router-based workflows where an LLM intelligently
    selects appropriate tools for each subtask.
    """

    def __init__(
        self,
        router_agent: BaseAgent,
        tasks: list[dict[str, Any]] | None = None,
        name: str = "RouterTask",
    ):
        """
        Initialize the router task.

        Args:
            router_agent: RouterAgent instance to handle task routing
            tasks: List of task dictionaries with 'description' and optional context
                  Each task dict can contain:
                  - description: Task description for the LLM
                  - image_path: Path to image file (optional)
                  - target_color: RGB color array (optional)
                  - Any other context needed for the task
            name: Task name
        """
        self.router_agent = router_agent
        self.tasks = tasks or []
        self.name = name

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute all tasks using the router agent synchronously.

        This is a wrapper around the async implementation.

        Args:
            context: Workflow context, may contain 'router_tasks'
            object_store: Storage for artifacts

        Returns:
            Updated context with router results
        """
        return asyncio.run(self.arun(context, object_store))

    async def arun(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute all tasks using the router agent asynchronously.

        If tasks are not provided during initialization, they can be passed
        in the context under the 'router_tasks' key.

        Args:
            context: Workflow context, may contain 'router_tasks'
            object_store: Storage for artifacts

        Returns:
            Updated context with:
            - router_results: List of results from each task
            - tasks_completed: Number of tasks completed
            - all_success: Whether all tasks succeeded
        """
        # Get tasks from initialization or context
        tasks_to_process = self.tasks or context.get("router_tasks", [])

        if not tasks_to_process:
            logger.warning("RouterTask has no tasks to process")
            context["router_results"] = []
            context["tasks_completed"] = 0
            context["all_success"] = True
            return context

        results = []

        for i, task_info in enumerate(tasks_to_process, 1):
            task_desc = task_info.get("description", f"Task {i}")
            logger.info(f"Processing task {i}/{len(tasks_to_process)}: {task_desc}")

            # Prepare task context with actual file paths and parameters
            task_context = {}

            # Add any other context from the task info
            for key, value in task_info.items():
                if key != "description":
                    task_context[key] = value

            # Execute with router asynchronously
            result = await self.router_agent.arun(
                task=task_desc, context=task_context, object_store=object_store
            )

            # Store result
            results.append(result)
            if object_store:
                object_store.set(f"task_{i}_result", result)

        # Update context with results
        context["router_results"] = results
        context["tasks_completed"] = len(results)
        context["all_success"] = all(r.get("success", False) for r in results)

        return context
