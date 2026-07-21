# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared types for Joint Agent API."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class APIResult:
    """Base result class for all API operations."""

    success: bool
    error: str | None = None


@dataclass
class MetricsResult:
    """Metrics result for predictions."""

    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0


@dataclass
class PipelineStepResult:
    """Result from a single pipeline step."""

    step_name: str
    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
