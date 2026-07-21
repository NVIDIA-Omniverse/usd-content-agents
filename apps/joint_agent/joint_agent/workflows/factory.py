# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow factory for Joint Agent.

This module provides factory functions to create workflows for different operations.
"""

import logging

from world_understanding.agentic import create_usd_dataset_workflow
from world_understanding.agentic.domain_tasks import ModelProvisioningTask
from world_understanding.agentic.usd_tasks import RenderScenePreviewTask
from world_understanding.agentic.workflows import Workflow
from world_understanding.utils.object_store import (
    InMemoryObjectStore,
    TempDirObjectStore,
)

from joint_agent.tasks.analyze_structure import AnalyzeStructureTask
from joint_agent.tasks.apply_joint_rigger import ApplyJointRiggerTask
from joint_agent.tasks.articulation_candidates import (
    AdjudicationModelProvisioningTask,
    ArticulationCandidatesTask,
)
from joint_agent.tasks.author_physics_schemas import AuthorPhysicsSchemasTask
from joint_agent.tasks.config_analyze_structure import AnalyzeStructureConfigTask
from joint_agent.tasks.config_apply_joint_rigger import ApplyJointRiggerConfigTask
from joint_agent.tasks.config_articulation_candidates import (
    ArticulationCandidatesConfigTask,
)
from joint_agent.tasks.config_author_physics_schemas import (
    AuthorPhysicsSchemasConfigTask,
)
from joint_agent.tasks.config_consistency import ConsistencyPassConfigTask
from joint_agent.tasks.config_identify_asset import IdentifyAssetConfigTask
from joint_agent.tasks.config_optimize_usd import OptimizeUSDConfigTask
from joint_agent.tasks.config_predict import PredictConfigTask
from joint_agent.tasks.config_prepare_dataset import PrepareDatasetConfigTask
from joint_agent.tasks.config_restore_usd import RestoreUSDConfigTask
from joint_agent.tasks.consistency import ConsistencyPassTask
from joint_agent.tasks.identify_asset import IdentifyAssetTask
from joint_agent.tasks.inference import VLMInferenceTask
from joint_agent.tasks.optimize_usd import OptimizeUSDTask
from joint_agent.tasks.predictions import SavePredictionsTask
from joint_agent.tasks.reporting import GeneratePredictionReportTask
from joint_agent.tasks.restore_usd import RestoreUSDTask

logger = logging.getLogger(__name__)


def create_prediction_workflow_from_config() -> Workflow:
    """Create a prediction workflow that loads config from file.

    The workflow:
    1. Loads configuration from YAML file
    2. Provisions VLM and LLM models
    3. Loads dataset
    4. Runs VLM inference on the dataset
    5. Generates a report
    6. Saves predictions

    Returns:
        Workflow instance ready for execution
    """
    # Import here to avoid circular imports
    from joint_agent.tasks.dataset_loading import DatasetLoadingTask

    return Workflow(
        tasks=[
            PredictConfigTask(),
            ModelProvisioningTask(),
            DatasetLoadingTask(),
            VLMInferenceTask(),
            GeneratePredictionReportTask(),
            SavePredictionsTask(),
        ],
        object_store=TempDirObjectStore(),
        name="Config-Driven Prediction",
        description="Run asset classification predictions",
    )


def create_usd_data_preparation_workflow_from_config() -> Workflow:
    """Create a USD data preparation workflow.

    The workflow:
    1. Loads configuration from YAML file
    2. Renders USD prims to images
    3. Creates dataset manifest

    Returns:
        Workflow instance ready for execution
    """
    return create_usd_dataset_workflow(
        workflow_name="USD → Dataset Preparation",
        workflow_description="Prepare prim→rendered views dataset from USD",
    )


def create_prepare_dataset_workflow_from_config() -> Workflow:
    """Create a dataset preparation workflow.

    The workflow:
    1. Loads configuration from YAML file
    2. Prepares dataset with prompts and metadata

    Note: No model provisioning needed - this workflow only formats dataset entries.

    Returns:
        Workflow instance ready for execution
    """
    # Import here to avoid circular imports
    from joint_agent.tasks.prepare_dataset import PrepareDatasetTask

    return Workflow(
        tasks=[
            PrepareDatasetConfigTask(),
            PrepareDatasetTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Prepare Dataset",
        description="Prepare dataset for asset classification",
    )


def create_identify_asset_workflow_from_config() -> Workflow:
    """Create an asset identification workflow.

    The workflow:
    1. Loads configuration and maps context keys for the common render task
    2. Renders lightweight whole-scene preview images (self-contained)
    3. Provisions VLM model
    4. Runs VLM inference to identify the whole asset

    Returns:
        Workflow instance ready for execution
    """
    return Workflow(
        tasks=[
            IdentifyAssetConfigTask(),
            RenderScenePreviewTask(),
            ModelProvisioningTask(),
            IdentifyAssetTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Asset Identification",
        description="Identify whole asset from preview renders",
    )


def create_optimize_usd_workflow_from_config() -> Workflow:
    """Create a USD optimization workflow.

    The workflow:
    1. Loads configuration from YAML file
    2. Optimizes USD file via NVCF API (mesh splitting, deduplication, etc.)

    Returns:
        Workflow instance ready for execution
    """
    return Workflow(
        tasks=[
            OptimizeUSDConfigTask(),
            OptimizeUSDTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven USD Optimization",
        description="Optimize USD file via NVCF Scene Optimizer API",
    )


def create_restore_usd_workflow_from_config() -> Workflow:
    """Create a predictions restoration workflow.

    The workflow:
    1. Loads configuration (original USD path, predictions path, optimization metadata)
    2. Transforms predictions from optimized USD prim paths back to original USD prim paths

    Returns:
        Workflow instance ready for execution
    """
    return Workflow(
        tasks=[
            RestoreUSDConfigTask(),
            RestoreUSDTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Predictions Restoration",
        description="Restore predictions from optimized to original USD structure",
    )


def create_consistency_pass_workflow_from_config() -> Workflow:
    """Create a prediction consistency post-processing workflow."""
    return Workflow(
        tasks=[
            ConsistencyPassConfigTask(),
            ConsistencyPassTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Consistency Pass",
        description="Post-process predictions for repeated-part consistency",
    )


def create_articulation_candidates_workflow_from_config() -> Workflow:
    """Create a Stage 2 articulation candidate inference workflow."""
    return Workflow(
        tasks=[
            ArticulationCandidatesConfigTask(),
            AdjudicationModelProvisioningTask(),
            ArticulationCandidatesTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Articulation Candidates",
        description="Infer report-only articulation candidates from predictions",
    )


def create_apply_joint_rigger_workflow_from_config() -> Workflow:
    """Create a Joint Rigger apply-step workflow."""
    return Workflow(
        tasks=[
            ApplyJointRiggerConfigTask(),
            ApplyJointRiggerTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Joint Rigger Apply",
        description="Run the Joint Rigger adapter boundary",
    )


def create_author_physics_schemas_workflow_from_config() -> Workflow:
    """Create the explicit post-rigger physics schema authoring workflow."""
    return Workflow(
        tasks=[
            AuthorPhysicsSchemasConfigTask(),
            AuthorPhysicsSchemasTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Physics Schema Authoring",
        description="Author explicit post-rigger physics schemas",
    )


def create_analyze_structure_workflow_from_config() -> Workflow:
    """Create a structure analysis workflow for articulated bodies.

    The workflow:
    1. Loads configuration (USD path, strategy, segment names)
    2. Provisions LLM model
    3. Analyzes hierarchy structure and produces segment assignments

    Returns:
        Workflow instance ready for execution
    """
    return Workflow(
        tasks=[
            AnalyzeStructureConfigTask(),
            ModelProvisioningTask(),
            AnalyzeStructureTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Structure Analysis",
        description="Analyze articulated body structure for segment assignment",
    )
