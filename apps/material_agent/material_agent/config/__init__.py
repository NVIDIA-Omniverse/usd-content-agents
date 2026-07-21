# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified configuration system for Material Agent.

This module provides a centralized configuration system where:
- All paths are auto-derived from project settings
- Single config format works for all commands
- No path duplication or manual wiring needed
"""

from material_agent.config.path_resolver import ProjectPathResolver
from material_agent.config.real_smoke_guardrails import (
    RealSmokeDisqualifier,
    RealSmokeGuardrailError,
    collect_real_smoke_disqualifiers,
    validate_real_smoke_guardrails,
)
from material_agent.config.unified_config import UnifiedPipelineConfigTask
from material_agent.config.validator import ConfigValidator

__all__ = [
    "ProjectPathResolver",
    "RealSmokeDisqualifier",
    "RealSmokeGuardrailError",
    "UnifiedPipelineConfigTask",
    "ConfigValidator",
    "collect_real_smoke_disqualifiers",
    "validate_real_smoke_guardrails",
]
