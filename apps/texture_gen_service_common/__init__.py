# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Texture Variation API service primitives."""

from .artifacts import local_file_uri, local_path_from_file_uri, require_visible_file
from .backend import (
    BackendHealth,
    TextureGenerationBackend,
    TextureGenerationBackendError,
)
from .models import (
    BackendCapabilities,
    Conditioning,
    Configuration,
    CreateJobRequest,
    GeneratedTextures,
    GenerationResult,
    HealthResponse,
    JobStatus,
    MapArtifact,
    TextureTarget,
)
from .prompting import (
    NIM_MAX_PROMPT_CHARS,
    PromptBudgetError,
    append_bounded_instruction,
)
from .service import (
    ServiceBusyError,
    ServiceNotReadyError,
    TextureVariationService,
    create_app,
)

__all__ = [
    "BackendCapabilities",
    "BackendHealth",
    "Conditioning",
    "Configuration",
    "CreateJobRequest",
    "GeneratedTextures",
    "GenerationResult",
    "HealthResponse",
    "JobStatus",
    "MapArtifact",
    "NIM_MAX_PROMPT_CHARS",
    "PromptBudgetError",
    "ServiceBusyError",
    "ServiceNotReadyError",
    "TextureGenerationBackend",
    "TextureGenerationBackendError",
    "TextureTarget",
    "TextureVariationService",
    "append_bounded_instruction",
    "create_app",
    "local_file_uri",
    "local_path_from_file_uri",
    "require_visible_file",
]
