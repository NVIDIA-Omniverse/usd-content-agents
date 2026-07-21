# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the unified pipeline workflow shim."""

from __future__ import annotations

from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import TempDirObjectStore

from material_agent.workflows.unified_pipeline import create_unified_pipeline_workflow


def test_create_unified_pipeline_workflow_builds_expected_tasks() -> None:
    workflow = create_unified_pipeline_workflow()

    assert isinstance(workflow, Workflow)
    assert workflow.name == "Unified Pipeline"
    assert workflow.description == (
        "Unified pipeline with auto-derived paths and single config format"
    )
    assert isinstance(workflow.object_store, TempDirObjectStore)
    assert [type(task).__name__ for task in workflow.tasks] == [
        "UnifiedPipelineConfigTask",
        "UnifiedPipelineExecutorTask",
    ]
