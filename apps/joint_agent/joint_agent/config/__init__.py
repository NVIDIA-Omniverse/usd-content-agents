# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration system for Joint Agent."""

from joint_agent.config.model_aliases import normalize_analyze_structure_model_alias
from joint_agent.config.path_resolver import ProjectPathResolver
from joint_agent.config.schema import (
    STEP_ORDER,
    STEP_OUTPUT_DIRS,
    get_default_config,
    get_step_defaults,
)
from joint_agent.config.unified_config import UnifiedPipelineConfigTask
from joint_agent.config.validator import ConfigValidator

__all__ = [
    "ConfigValidator",
    "ProjectPathResolver",
    "STEP_ORDER",
    "STEP_OUTPUT_DIRS",
    "UnifiedPipelineConfigTask",
    "get_default_config",
    "get_step_defaults",
    "normalize_analyze_structure_model_alias",
]
