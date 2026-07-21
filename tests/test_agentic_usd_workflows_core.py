# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic import usd_workflows
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import InMemoryObjectStore


def test_create_usd_dataset_workflow_builds_standard_task_order(caplog):
    caplog.set_level(logging.DEBUG, logger=usd_workflows.__name__)

    workflow = usd_workflows.create_usd_dataset_workflow(
        workflow_name="Custom USD Workflow",
        workflow_description="Custom description",
    )

    assert isinstance(workflow, Workflow)
    assert isinstance(workflow.object_store, InMemoryObjectStore)
    assert workflow.name == "Custom USD Workflow"
    assert workflow.description == "Custom description"
    assert [task.__class__.__name__ for task in workflow.tasks] == [
        "USDDataPrepConfigTask",
        "USDRendererProvisioningTask",
        "USDLoadingTask",
        "USDPrimTraversalAndRenderingTask",
        "USDDatasetManifestTask",
    ]
    assert "Created USD dataset workflow with 5 tasks" in caplog.text


def test_run_usd_dataset_workflow_passes_config_and_overrides(monkeypatch):
    calls: list[dict[str, Any]] = []

    class FakeWorkflow:
        def run(self, initial_context: dict[str, Any]) -> dict[str, Any]:
            calls.append(initial_context)
            return {"dataset_path": "dataset.json", "initial_context": initial_context}

    def fake_create_usd_dataset_workflow(**kwargs: Any) -> FakeWorkflow:
        calls.append(kwargs)
        return FakeWorkflow()

    monkeypatch.setattr(
        usd_workflows,
        "create_usd_dataset_workflow",
        fake_create_usd_dataset_workflow,
    )

    result = usd_workflows.run_usd_dataset_workflow(
        Path("config.yaml"),
        overrides={"batch_size": 4},
    )

    assert calls == [
        {"config_path": Path("config.yaml"), "overrides": {"batch_size": 4}},
        {"config_path": Path("config.yaml"), "batch_size": 4},
    ]
    assert result["dataset_path"] == "dataset.json"
