# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded texture workflow contracts and mock adapters."""

from .client import (
    MockTextureExecutionCall,
    MockTexturePlannerExecutorClient,
    TextureAgentServiceClient,
    TexturePlannerExecutorClient,
)
from .finalizer import (
    CanonicalTextureWorkflowFinalizer,
    TextureWorkflowFinalizer,
    write_texture_planning_artifacts,
)
from .models import (
    TEXTURE_FINALIZATION_RESULT_SCHEMA_VERSION,
    TEXTURE_FINALIZER_INPUT_SCHEMA_VERSION,
    TEXTURE_PLAN_SCHEMA_VERSION,
    TEXTURE_VALIDATION_EVIDENCE_SCHEMA_VERSION,
    TEXTURE_WORKFLOW_PROGRESS_SCHEMA_VERSION,
    TextureExecutionResult,
    TextureFinalizationResult,
    TextureFinalizerInput,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanDocument,
    TexturePlanSelectedUnit,
    TextureUnitArtifact,
    TextureValidationFinding,
    TextureValidationResult,
    TextureWorkflowProgress,
    TextureWorkflowRequest,
    TextureWorkflowValidationEvidence,
)
from .workbench_validation import (
    MockWorkbenchTextureValidator,
    MockWorkbenchValidationCall,
    TextureWorkbenchValidator,
)
from .workflow import (
    ProgressCallback,
    run_batch_texture_workflow,
    run_interactive_texture_workflow,
    run_texture_workflow,
)

__all__ = [
    "CanonicalTextureWorkflowFinalizer",
    "MockTextureExecutionCall",
    "MockTexturePlannerExecutorClient",
    "MockWorkbenchTextureValidator",
    "MockWorkbenchValidationCall",
    "ProgressCallback",
    "TEXTURE_FINALIZATION_RESULT_SCHEMA_VERSION",
    "TEXTURE_FINALIZER_INPUT_SCHEMA_VERSION",
    "TEXTURE_PLAN_SCHEMA_VERSION",
    "TEXTURE_VALIDATION_EVIDENCE_SCHEMA_VERSION",
    "TEXTURE_WORKFLOW_PROGRESS_SCHEMA_VERSION",
    "TextureAgentServiceClient",
    "TextureExecutionResult",
    "TextureFinalizationResult",
    "TextureFinalizerInput",
    "TexturePlanCounts",
    "TexturePlanDecision",
    "TexturePlanDocument",
    "TexturePlanSelectedUnit",
    "TexturePlannerExecutorClient",
    "TextureUnitArtifact",
    "TextureValidationFinding",
    "TextureValidationResult",
    "TextureWorkbenchValidator",
    "TextureWorkflowFinalizer",
    "TextureWorkflowProgress",
    "TextureWorkflowRequest",
    "TextureWorkflowValidationEvidence",
    "run_batch_texture_workflow",
    "run_interactive_texture_workflow",
    "run_texture_workflow",
    "write_texture_planning_artifacts",
]
