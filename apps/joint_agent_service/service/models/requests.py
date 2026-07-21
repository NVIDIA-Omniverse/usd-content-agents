# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request models for Joint Agent Service API."""

from enum import StrEnum

from pydantic import BaseModel, Field


class PipelineStep(StrEnum):
    """Available pipeline steps."""

    OPTIMIZE_USD = "optimize_usd"
    IDENTIFY_ASSET = "identify_asset"
    ANALYZE_STRUCTURE = "analyze_structure"
    BUILD_DATASET_USD = "build_dataset_usd"
    BUILD_DATASET_PREPARE_DATASET = "build_dataset_prepare_dataset"
    PREDICT = "predict"
    CONSISTENCY_PASS = "consistency_pass"
    INFER_ARTICULATION_CANDIDATES = "infer_articulation_candidates"
    RESTORE_USD = "restore_usd"


class RegenerateRequest(BaseModel):
    """Request to regenerate specific steps from cache."""

    steps: list[PipelineStep] = Field(description="Steps to re-run from cache")

    user_prompt: str | None = Field(
        default=None, description="Override user prompt for regeneration"
    )
