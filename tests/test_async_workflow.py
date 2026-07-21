# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async workflow execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from world_understanding.agentic import workflows as workflows_module
from world_understanding.agentic.tasks import CallableTask, Task
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import InMemoryObjectStore, ObjectStore
from world_understanding.utils.result_projection import project_result_metadata


class SimpleTask(Task):
    """A simple test task."""

    def __init__(self, name: str, value_to_add: int):
        self.name = name
        self.value_to_add = value_to_add

    def run(self, context, object_store=None):
        """Synchronous run - should delegate to async."""
        return asyncio.run(self.arun(context, object_store))

    async def arun(self, context, object_store=None):
        """Async implementation."""
        # Simulate some async work
        await asyncio.sleep(0.01)

        # Update context
        count = context.get("count", 0)
        context["count"] = count + self.value_to_add
        context[f"{self.name}_executed"] = True

        return context


class _RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.statuses: list[Any] = []
        self.exceptions: list[Exception] = []

    def __enter__(self) -> _RecordingSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        self.statuses.append(status)

    def record_exception(self, exception: Exception) -> None:
        self.exceptions.append(exception)


class _RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def start_as_current_span(self, name: str) -> _RecordingSpan:
        span = _RecordingSpan(name)
        self.spans.append(span)
        return span


class _FailingTask(Task):
    name = "FailingTask"

    def __init__(self, message: str) -> None:
        self.message = message

    def run(self, context, object_store=None):
        raise RuntimeError(self.message)

    async def arun(self, context, object_store=None):
        raise RuntimeError(self.message)


class _CredentialNamedTask(Task):
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.dispatch_names: list[str] = []

    def run(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(self.arun(context, object_store))

    async def arun(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        self.dispatch_names.append(self.name)
        if self.fail:
            raise RuntimeError("task failed")
        context["task_ran"] = True
        return context


def test_workflow_sync_execution():
    """Test that sync workflow execution still works."""
    # Create workflow with tasks
    workflow = Workflow(
        tasks=[
            SimpleTask("task1", 1),
            SimpleTask("task2", 2),
            SimpleTask("task3", 3),
        ],
        name="TestWorkflow",
    )

    # Run workflow synchronously
    result = workflow.run({"count": 0})

    # Verify results
    assert result["count"] == 6  # 1 + 2 + 3
    assert result["task1_executed"] is True
    assert result["task2_executed"] is True
    assert result["task3_executed"] is True
    assert result["workflow_completed"] is True


def test_workflow_add_and_clear_tasks():
    workflow = Workflow(name="MutableWorkflow")
    task = SimpleTask("task", 1)

    workflow.add_task(task)
    assert workflow.tasks == [task]

    workflow.clear_tasks()
    assert workflow.tasks == []


@pytest.mark.asyncio
async def test_workflow_async_execution():
    """Test that async workflow execution works."""
    # Create workflow with tasks
    workflow = Workflow(
        tasks=[
            SimpleTask("task1", 1),
            SimpleTask("task2", 2),
            SimpleTask("task3", 3),
        ],
        name="TestWorkflow",
    )

    # Run workflow asynchronously
    result = await workflow.arun({"count": 0})

    # Verify results
    assert result["count"] == 6  # 1 + 2 + 3
    assert result["task1_executed"] is True
    assert result["task2_executed"] is True
    assert result["task3_executed"] is True
    assert result["workflow_completed"] is True


@pytest.mark.asyncio
async def test_workflow_early_termination():
    """Test that workflow early termination works in async mode."""

    class TerminatingTask(Task):
        """A task that terminates the workflow."""

        def __init__(self, name: str):
            self.name = name

        def run(self, context, object_store=None):
            return asyncio.run(self.arun(context, object_store))

        async def arun(self, context, object_store=None):
            context[f"{self.name}_executed"] = True
            context["workflow_terminated"] = True
            return context

    # Create workflow
    workflow = Workflow(
        tasks=[
            SimpleTask("task1", 1),
            TerminatingTask("terminator"),
            SimpleTask("task2", 2),  # Should not execute
        ],
        name="TestWorkflow",
    )

    # Run workflow
    result = await workflow.arun({"count": 0})

    # Verify results
    assert result["count"] == 1  # Only task1 executed
    assert result["task1_executed"] is True
    assert result["terminator_executed"] is True
    assert result.get("task2_executed") is None  # Should not have executed
    assert result["workflow_completed"] is False


@pytest.mark.asyncio
async def test_callable_task_async():
    """Test that CallableTask works asynchronously."""

    def my_function(context, object_store):
        context["called"] = True
        context["value"] = 42
        return context

    task = CallableTask(my_function, name="MyCallable")

    # Test async execution
    result = await task.arun({})

    assert result["called"] is True
    assert result["value"] == 42


def test_workflow_with_object_store():
    """Test that object store works with async workflow."""

    class StoreTask(Task):
        """A task that uses object store."""

        def __init__(self, name: str, key: str, value: str):
            self.name = name
            self.key = key
            self.value = value

        def run(self, context, object_store=None):
            return asyncio.run(self.arun(context, object_store))

        async def arun(self, context, object_store=None):
            if object_store:
                object_store.set(self.key, self.value)
                context[f"{self.name}_stored"] = True
            return context

    # Create workflow with object store
    object_store = InMemoryObjectStore()
    workflow = Workflow(
        tasks=[
            StoreTask("task1", "key1", "value1"),
            StoreTask("task2", "key2", "value2"),
        ],
        object_store=object_store,
        name="StoreWorkflow",
    )

    # Run workflow
    result = workflow.run({})

    # Verify results
    assert result["task1_stored"] is True
    assert result["task2_stored"] is True
    assert object_store.get("key1") == "value1"
    assert object_store.get("key2") == "value2"


def test_workflow_failure_diagnostics_are_value_free_for_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinels = (
        "never-log-workflow-config-713",
        "never-log-workflow-source-713",
        "never-log-workflow-output-713",
    )
    tracer = _RecordingTracer()
    monkeypatch.setattr(workflows_module, "get_tracer", lambda _name: tracer)

    results: list[dict[str, Any]] = []
    with caplog.at_level(logging.INFO, logger=workflows_module.__name__):
        for mode in ("sync", "async"):
            initial_context = {
                "config_path": f"/{sentinels[0]}/config.yaml",
                "source_override": f"/{sentinels[1]}/scene.usd",
                "output_dir_override": f"/{sentinels[2]}",
            }
            workflow = Workflow(
                tasks=[_FailingTask("backend reflected " + " ".join(sentinels))],
                name=f"DiagnosticBoundary-{mode}",
            )
            result = (
                workflow.run(initial_context)
                if mode == "sync"
                else asyncio.run(workflow.arun(initial_context))
            )
            results.append(result)

            assert result["config_path"] == initial_context["config_path"]
            assert result["source_override"] == initial_context["source_override"]
            assert (
                result["output_dir_override"] == initial_context["output_dir_override"]
            )
            assert result["error"] == "Task execution failed"
            assert result["failed_task"] == "FailingTask"
            assert result["workflow_completed"] is False

    task_spans = [span for span in tracer.spans if span.name.startswith("task.")]
    assert len(task_spans) == 2
    assert all(not span.exceptions for span in task_spans)
    assert all(len(span.statuses) == 1 for span in task_spans)
    assert all(
        status.description == "Task execution failed"
        for span in task_spans
        for status in span.statuses
    )
    diagnostic_surfaces = (
        caplog.text
        + repr([span.name for span in tracer.spans])
        + repr([span.attributes for span in tracer.spans])
        + repr([span.statuses for span in tracer.spans])
        + repr([result["error"] for result in results])
    )
    for sentinel in sentinels:
        assert sentinel not in diagnostic_surfaces


@pytest.mark.parametrize("run_sync", [True, False])
@pytest.mark.parametrize("fail", [False, True])
def test_workflow_projects_credentialed_names_but_dispatches_raw_task(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    run_sync: bool,
    fail: bool,
) -> None:
    secret = "workflow-name-secret-713"
    raw_name = f"https://runtime-user:{secret}@workflow.example.test/run"
    task = _CredentialNamedTask(raw_name, fail=fail)
    tracer = _RecordingTracer()
    monkeypatch.setattr(workflows_module, "get_tracer", lambda _name: tracer)
    workflow = Workflow(tasks=[task], name=raw_name)

    with caplog.at_level(logging.INFO, logger=workflows_module.__name__):
        result = workflow.run({}) if run_sync else asyncio.run(workflow.arun({}))

    assert task.dispatch_names == [raw_name]
    assert result["workflow_name"] == raw_name
    assert result["current_task"] == raw_name
    if fail:
        assert result["failed_task"] == raw_name
        assert result["workflow_completed"] is False
    else:
        assert result["task_ran"] is True
        assert result["workflow_completed"] is True

    published_result = project_result_metadata(result)
    assert published_result["workflow_name"] == "<redacted>"
    assert published_result["current_task"] == "<redacted>"
    if fail:
        assert published_result["failed_task"] == "<redacted>"

    observable = (
        f"{published_result!r}\n{caplog.text}\n"
        f"{[span.name for span in tracer.spans]!r}\n"
        f"{[span.attributes for span in tracer.spans]!r}"
    )
    assert raw_name not in observable
    assert secret not in observable
