# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render ``RobotPromptEntry`` objects into LLM-ready prompt strings.

Two rendering targets:
  * Segment-inference system prompt — augments
    ``INFER_SEGMENTS_SYSTEM_PROMPT`` with the robot's known DOFs and component
    names so the model returns the canonical taxonomy verbatim.
  * Hierarchy-analysis system prompt — augments ``ANALYSIS_SYSTEM_PROMPT``
    with per-component visual cues and tiebreakers, mirroring the verbose
    UR10e prompt that worked end-to-end on a stripped Blender variant.
"""

from __future__ import annotations

from joint_agent.prompts.schema import RobotPromptEntry


def render_segment_inference_system_prompt(
    base_prompt: str, entry: RobotPromptEntry
) -> str:
    """Augment the segment-inference system prompt with entry-specific facts."""
    lines = [
        base_prompt.rstrip(),
        "",
        f"## Known robot: {entry.description}",
        f"- topology: {entry.topology}",
        f"- DOFs: {entry.num_dofs}",
        f"- segments ({len(entry.component_names)}, base to end-effector):",
        "    " + ", ".join(entry.component_names),
        "",
        (
            "Use these exact segment names verbatim in your response — they "
            "define the constrained taxonomy expected downstream."
        ),
    ]
    return "\n".join(lines)


def render_analysis_system_prompt(base_prompt: str, entry: RobotPromptEntry) -> str:
    """Augment the hierarchy-analysis system prompt with entry-specific cues."""
    lines = [base_prompt.rstrip(), ""]
    lines.append(f"## Robot context: {entry.description}")
    lines.append(
        f"This asset is a {entry.topology.replace('_', ' ')} with "
        f"{entry.num_dofs} DOFs and {len(entry.component_names)} segments."
    )
    lines.append("")

    lines.append("## Required component_name values (use exactly these strings):")
    for name in entry.component_names:
        lines.append(f"  - {name}")
    lines.append("")

    lines.append("## Visual identification cues (per segment):")
    for name in entry.component_names:
        cue = entry.visual_cues[name]
        lines.append(f"  - {name}: {cue}")
    lines.append("")

    if entry.tiebreakers:
        lines.append("## Tiebreakers for ambiguous cases:")
        for rule in entry.tiebreakers:
            lines.append(f"  - {rule}")
        lines.append("")

    if entry.completeness_required:
        lines.append(
            "## Completeness requirement: every component_name listed above "
            "must appear at least once in your output. Multiple meshes may "
            "share the same component_name (a physical link is often made of "
            "several mesh prims), but no role may be entirely absent."
        )
        lines.append("")

    if entry.self_check:
        lines.append(
            "## Self-check: after assigning roles, verify every "
            "component_name listed above appears in your output. If any role "
            "is missing, identify the assignment you are least confident "
            "about and reconsider whether that prim should carry the missing "
            "role instead."
        )

    return "\n".join(lines).rstrip() + "\n"
