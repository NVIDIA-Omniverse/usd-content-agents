# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified pipeline workflow for Joint Agent.

This module provides the main orchestrator for multi-step pipelines.
"""

import logging

from world_understanding.agentic.workflows import Workflow

from joint_agent.config.unified_config import UnifiedPipelineConfigTask
from joint_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask

logger = logging.getLogger(__name__)


def create_unified_pipeline_workflow() -> Workflow:
    """Create the unified pipeline workflow.

    This workflow:
    1. Loads unified configuration
    2. Resolves all paths automatically
    3. Executes requested steps in dependency order
    4. Manages data flow between steps

    Returns:
        Workflow instance ready for execution
    """
    return Workflow(
        tasks=[
            UnifiedPipelineConfigTask(),
            UnifiedPipelineExecutorTask(),
        ]
    )
