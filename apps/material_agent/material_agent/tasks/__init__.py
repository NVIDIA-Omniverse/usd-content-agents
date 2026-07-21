# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material Agent tasks for workflow composition."""

from world_understanding.agentic.domain_tasks import ModelProvisioningTask
from world_understanding.agentic.usd_tasks import (
    OptimizeUSDConfigTask,
    OptimizeUSDTask,
    RestoreUSDConfigTask,
    RestoreUSDTask,
)

from material_agent.tasks.apply_completion import ApplyCompletionTask
from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask
from material_agent.tasks.cluster_prims import (
    ClusterPrimsTask,
    ExpandClusterPredictionsTask,
)
from material_agent.tasks.config_apply import ApplyConfigTask
from material_agent.tasks.config_benchmark import BenchmarkConfigTask
from material_agent.tasks.config_cluster_prims import (
    ClusterPrimsConfigTask,
    ExpandClusterPredictionsConfigTask,
)
from material_agent.tasks.config_evaluate import EvaluateConfigTask
from material_agent.tasks.config_generate import GenerateConfigTask
from material_agent.tasks.config_iterative_apply import IterativeApplyConfigTask
from material_agent.tasks.config_pdf_vectorstore import PDFVectorstoreConfigTask
from material_agent.tasks.config_pipeline import PipelineConfigTask
from material_agent.tasks.config_predict import PredictConfigTask
from material_agent.tasks.config_prepare_dataset import PrepareDatasetConfigTask
from material_agent.tasks.dataset import DatasetLoadingTask
from material_agent.tasks.evaluation import EvaluationTask
from material_agent.tasks.generate_material_library import GenerateMaterialLibraryTask
from material_agent.tasks.generate_material_library_config import (
    GenerateMaterialLibraryConfigTask,
)
from material_agent.tasks.generate_ref_image_config import GenerateRefImageConfigTask
from material_agent.tasks.identify_materials import IdentifyUniqueMaterialsTask
from material_agent.tasks.inference import VLMInferenceTask
from material_agent.tasks.iteration import IterationTask
from material_agent.tasks.iterative_completion import IterativeApplyCompletionTask
from material_agent.tasks.judge import JudgeTask
from material_agent.tasks.material_creation_policy import MaterialDecisionPolicyTask
from material_agent.tasks.material_retrieval import MaterialRetrievalTask
from material_agent.tasks.predictions import SavePredictionsTask
from material_agent.tasks.prepare_dataset import PrepareDatasetTask
from material_agent.tasks.render import RenderTask
from material_agent.tasks.render_config import RenderConfigTask
from material_agent.tasks.render_preview_config import RenderPreviewConfigTask
from material_agent.tasks.reporting import (
    GenerateEvaluationReportTask,
    GeneratePredictionReportTask,
)
from material_agent.tasks.resolve_materials import ResolveMaterialFilesTask
from material_agent.tasks.self_evaluation import SelfEvaluationTask
from material_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask
from material_agent.tasks.visual_grounding import VisualGroundingTask

__all__ = [
    "ApplyCompletionTask",
    "ApplyConfigTask",
    "ApplyMaterialsToUSDTask",
    "BenchmarkConfigTask",
    "ClusterPrimsTask",
    "ClusterPrimsConfigTask",
    "DatasetLoadingTask",
    "EvaluateConfigTask",
    "EvaluationTask",
    "ExpandClusterPredictionsTask",
    "ExpandClusterPredictionsConfigTask",
    "GenerateConfigTask",
    "GenerateEvaluationReportTask",
    "GenerateMaterialLibraryConfigTask",
    "GenerateMaterialLibraryTask",
    "GeneratePredictionReportTask",
    "GenerateRefImageConfigTask",
    "IdentifyUniqueMaterialsTask",
    "IterationTask",
    "IterativeApplyCompletionTask",
    "IterativeApplyConfigTask",
    "JudgeTask",
    "MaterialDecisionPolicyTask",
    "MaterialRetrievalTask",
    "ModelProvisioningTask",
    "OptimizeUSDConfigTask",
    "OptimizeUSDTask",
    "PDFVectorstoreConfigTask",
    "PipelineConfigTask",
    "PrepareDatasetTask",
    "PredictConfigTask",
    "PrepareDatasetConfigTask",
    "RenderConfigTask",
    "RenderPreviewConfigTask",
    "RenderTask",
    "ResolveMaterialFilesTask",
    "RestoreUSDConfigTask",
    "RestoreUSDTask",
    "SavePredictionsTask",
    "SelfEvaluationTask",
    "UnifiedPipelineExecutorTask",
    "VisualGroundingTask",
    "VLMInferenceTask",
]
