# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Joint Agent Service request models."""

from apps.joint_agent_service.service.models.requests import (
    PipelineStep,
    RegenerateRequest,
)


def test_regenerate_request_accepts_pipeline_steps() -> None:
    step_values = [step.value for step in PipelineStep]

    request = RegenerateRequest.model_validate({"steps": step_values})

    assert request.steps == list(PipelineStep)
