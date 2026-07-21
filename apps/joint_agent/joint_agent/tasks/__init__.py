# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent Tasks."""

from joint_agent.tasks.apply_joint_rigger import ApplyJointRiggerTask
from joint_agent.tasks.articulation_candidates import (
    AdjudicationModelProvisioningTask,
    ArticulationCandidatesTask,
)
from joint_agent.tasks.author_physics_schemas import AuthorPhysicsSchemasTask
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
from joint_agent.tasks.config_usd_dataset import USDDatasetConfigTask
from joint_agent.tasks.consistency import ConsistencyPassTask
from joint_agent.tasks.dataset_loading import DatasetLoadingTask
from joint_agent.tasks.identify_asset import IdentifyAssetTask
from joint_agent.tasks.inference import VLMInferenceTask
from joint_agent.tasks.optimize_usd import OptimizeUSDTask
from joint_agent.tasks.predictions import SavePredictionsTask
from joint_agent.tasks.prepare_dataset import PrepareDatasetTask
from joint_agent.tasks.reporting import GeneratePredictionReportTask
from joint_agent.tasks.restore_usd import RestoreUSDTask
from joint_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask

__all__ = [
    "ApplyJointRiggerConfigTask",
    "ApplyJointRiggerTask",
    "ConsistencyPassConfigTask",
    "ConsistencyPassTask",
    "ArticulationCandidatesConfigTask",
    "AdjudicationModelProvisioningTask",
    "ArticulationCandidatesTask",
    "AuthorPhysicsSchemasConfigTask",
    "AuthorPhysicsSchemasTask",
    "IdentifyAssetConfigTask",
    "IdentifyAssetTask",
    "OptimizeUSDConfigTask",
    "OptimizeUSDTask",
    "PredictConfigTask",
    "PrepareDatasetConfigTask",
    "USDDatasetConfigTask",
    "DatasetLoadingTask",
    "VLMInferenceTask",
    "SavePredictionsTask",
    "PrepareDatasetTask",
    "GeneratePredictionReportTask",
    "RestoreUSDConfigTask",
    "RestoreUSDTask",
    "UnifiedPipelineExecutorTask",
]
