# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow factory for Physics Agent.

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

from physics_agent.tasks.apply_physics import ApplyPhysicsTask
from physics_agent.tasks.config_apply_physics import ApplyPhysicsConfigTask
from physics_agent.tasks.config_identify_asset import IdentifyAssetConfigTask
from physics_agent.tasks.config_optimize_usd import OptimizeUSDConfigTask
from physics_agent.tasks.config_predict import PredictConfigTask
from physics_agent.tasks.config_prepare_dataset import PrepareDatasetConfigTask
from physics_agent.tasks.config_restore_usd import RestoreUSDConfigTask
from physics_agent.tasks.config_vomp_mass import VompMassConfigTask
from physics_agent.tasks.identify_asset import IdentifyAssetTask
from physics_agent.tasks.inference import VLMInferenceTask
from physics_agent.tasks.optimize_usd import OptimizeUSDTask
from physics_agent.tasks.predictions import SavePredictionsTask
from physics_agent.tasks.reporting import GeneratePredictionReportTask
from physics_agent.tasks.restore_usd import RestoreUSDTask
from physics_agent.tasks.vomp_mass import VompMassTask

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
    from physics_agent.tasks.dataset_loading import DatasetLoadingTask

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
    from physics_agent.tasks.prepare_dataset import PrepareDatasetTask

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


def create_apply_physics_workflow_from_config() -> Workflow:
    """Create an apply-physics workflow.

    The workflow:
    1. Loads configuration (input USD, predictions JSONL, output path, collision approx)
    2. Applies UsdPhysics schemas (RigidBodyAPI, CollisionAPI, MassAPI, MaterialAPI)
       to each prim referenced in the predictions, writing a simulation-ready USD file

    Returns:
        Workflow instance ready for execution
    """
    return Workflow(
        tasks=[
            ApplyPhysicsConfigTask(),
            ApplyPhysicsTask(),
        ],
        object_store=InMemoryObjectStore(),
        name="Config-Driven Apply Physics",
        description="Apply UsdPhysics schemas from predictions to USD stage",
    )


def create_vomp_mass_workflow_from_config() -> Workflow:
    """Create the concrete OVRTX-to-VoMP mass-inference workflow."""

    return Workflow(
        tasks=[VompMassConfigTask(), VompMassTask()],
        object_store=InMemoryObjectStore(),
        name="Config-Driven VoMP Mass Inference",
        description="Render with OVRTX, infer with VoMP, and author MassAPI",
    )
