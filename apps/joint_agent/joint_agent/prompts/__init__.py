# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-robot system-prompt library for the joint agent.

Loads structured prompt definitions from YAML files in ``data/`` and exposes
helpers to resolve a robot ID from upstream asset-identification output and
render the entry into segment names and an augmented system prompt.
"""

from joint_agent.prompts.library import list_robots, load_prompt
from joint_agent.prompts.matching import lookup_robot_id
from joint_agent.prompts.prop_articulation import (
    PROP_ROLE_CARDS,
    SUPPORTED_PROP_ROLES,
    PropRolePromptCard,
    render_prop_articulation_system_prompt,
    render_prop_articulation_user_prompt,
)
from joint_agent.prompts.rendering import (
    render_analysis_system_prompt,
    render_segment_inference_system_prompt,
)
from joint_agent.prompts.schema import RobotPromptEntry

__all__ = [
    "PROP_ROLE_CARDS",
    "SUPPORTED_PROP_ROLES",
    "PropRolePromptCard",
    "RobotPromptEntry",
    "list_robots",
    "load_prompt",
    "lookup_robot_id",
    "render_analysis_system_prompt",
    "render_prop_articulation_system_prompt",
    "render_prop_articulation_user_prompt",
    "render_segment_inference_system_prompt",
]
