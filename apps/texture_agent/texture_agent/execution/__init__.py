# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public bounded-execution interfaces for Texture Plan consumers."""

from texture_agent.execution.adapters import bind_prim_texture_units_to_plan
from texture_agent.execution.executor import (
    BoundedTextureExecutor,
    CancellationToken,
    FileTextureExecutionCheckpointStore,
    TextureExecutionCancelled,
    TextureExecutionCheckpointStore,
    TextureExecutionTimedOut,
    TextureUnitExecutionContext,
    texture_plan_fingerprint,
)
from texture_agent.execution.models import (
    TEXTURE_EXECUTION_CHECKPOINT_SCHEMA_VERSION,
    TEXTURE_EXECUTION_SUMMARY_SCHEMA_VERSION,
    TextureArtifactRef,
    TextureExecutionCheckpoint,
    TextureExecutionStatus,
    TextureExecutionSummary,
    TextureUnitExecutionRecord,
    TextureUnitExecutionResult,
    TextureUnitExecutionState,
)

__all__ = [
    "TEXTURE_EXECUTION_CHECKPOINT_SCHEMA_VERSION",
    "TEXTURE_EXECUTION_SUMMARY_SCHEMA_VERSION",
    "BoundedTextureExecutor",
    "CancellationToken",
    "FileTextureExecutionCheckpointStore",
    "TextureArtifactRef",
    "TextureExecutionCancelled",
    "TextureExecutionCheckpoint",
    "TextureExecutionCheckpointStore",
    "TextureExecutionStatus",
    "TextureExecutionSummary",
    "TextureExecutionTimedOut",
    "TextureUnitExecutionContext",
    "TextureUnitExecutionRecord",
    "TextureUnitExecutionResult",
    "TextureUnitExecutionState",
    "bind_prim_texture_units_to_plan",
    "texture_plan_fingerprint",
]
