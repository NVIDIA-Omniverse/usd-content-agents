# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow orchestration for task execution."""

import asyncio
import logging
from typing import Any

from opentelemetry.trace import Status, StatusCode

from world_understanding.agentic.tasks import Task
from world_understanding.telemetry import get_tracer
from world_understanding.utils.credentials import redact_sensitive_path
from world_understanding.utils.model_auth import public_model_failure_message
from world_understanding.utils.object_store import (
    InMemoryObjectStore,
    ObjectStore,
)

logger = logging.getLogger(__name__)

_TASK_FAILURE_MESSAGE = "Task execution failed"


def _diagnostic_name(value: Any) -> str:
    """Project a runtime workflow identifier onto observable surfaces."""
    if not isinstance(value, str):
        return "<unavailable>"
    return redact_sensitive_path(value)


class Workflow:
    """
    Workflow orchestrator that executes tasks in sequence.

    More complex patterns (parallel execution, conditional branching) can be added as needed.
    """

    def __init__(
        self,
        tasks: list[Task] | None = None,
        object_store: ObjectStore | None = None,
        name: str = "Workflow",
        description: str = "",
    ):
        """
        Initialize the workflow.

        Args:
            tasks: List of tasks to execute in order
            object_store: Storage for artifacts (creates InMemoryObjectStore if None)
            name: Workflow name
            description: Workflow description
        """
        self.tasks = tasks or []
        self.name = name
        self.description = description

        if object_store is None:
            object_store = InMemoryObjectStore()
        self.object_store = object_store

    def run(self, initial_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the workflow synchronously.

        This is a wrapper around the async implementation for backward
        compatibility.

        Args:
            initial_context: Initial context for the workflow

        Returns:
            Final context after all tasks have executed
        """
        return asyncio.run(self.arun(initial_context))

    async def arun(
        self, initial_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute the workflow asynchronously.

        This is the core implementation. The sync run() delegates to this.

        Args:
            initial_context: Initial context for the workflow

        Returns:
            Final context after all tasks have executed
        """
        tracer = get_tracer(__name__)
        safe_workflow_name = _diagnostic_name(self.name)
        with tracer.start_as_current_span(
            f"workflow.{safe_workflow_name}"
        ) as workflow_span:
            context = initial_context or {}
            context["workflow_name"] = self.name
            workflow_span.set_attribute("workflow.name", safe_workflow_name)
            workflow_span.set_attribute("workflow.task_count", len(self.tasks))

            for i, task in enumerate(self.tasks):
                task_name = getattr(task, "name", task.__class__.__name__)
                safe_task_name = _diagnostic_name(task_name)
                context["current_task"] = task_name
                context["task_index"] = i

                logger.info(
                    "Executing task %d/%d: %s",
                    i + 1,
                    len(self.tasks),
                    safe_task_name,
                )

                with tracer.start_as_current_span(
                    f"task.{safe_task_name}"
                ) as task_span:
                    task_span.set_attribute("task.name", safe_task_name)
                    task_span.set_attribute("task.index", i)

                    try:
                        # Execute task asynchronously
                        context = await task.arun(context, self.object_store)

                        # Check for early termination
                        if context.get("workflow_terminated", False):
                            logger.info(
                                "Workflow terminated early at task %s",
                                safe_task_name,
                            )
                            break

                    except Exception as error:
                        # Task exceptions may contain config values, paths, or
                        # backend payloads. Do not copy exception text, stack
                        # frames, or causes into telemetry, logs, or context.
                        safe_error = public_model_failure_message(
                            error, _TASK_FAILURE_MESSAGE
                        )
                        task_span.set_status(Status(StatusCode.ERROR, safe_error))
                        logger.error("Task %s failed: %s", safe_task_name, safe_error)
                        context["error"] = safe_error
                        context["failed_task"] = task_name
                        context["workflow_terminated"] = True
                        break

            context["workflow_completed"] = not context.get(
                "workflow_terminated", False
            )
            return context

    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        self.tasks.append(task)

    def clear_tasks(self) -> None:
        """Clear all tasks from the workflow."""
        self.tasks.clear()
