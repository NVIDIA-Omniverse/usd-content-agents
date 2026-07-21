# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic schema for a per-robot prompt library entry."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Topology = Literal["serial_chain", "branching", "mobile_base", "humanoid"]


class RobotPromptEntry(BaseModel):
    """Structured prompt entry describing a single robot type.

    The ``component_names`` list is the constrained taxonomy emitted to
    downstream consumers (CIP phrase-rule templates, taxonomy normalization).
    Both ``component_names`` and ``visual_cues`` are authored together so
    ``visual_cues`` keys cover every component name (enforced by validator).
    """

    model_config = ConfigDict(extra="forbid")

    robot_id: str = Field(
        description="Stable identifier (kebab/snake_case). Matches YAML filename."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Alternative names (case-insensitive) used to resolve this entry "
            "from upstream asset_type/asset_subtype strings or USD filenames."
        ),
    )
    filename_aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Model-specific filename stems or stem fragments allowed for "
            "USD-path matching. Keep broad vendor/family names in aliases only."
        ),
    )
    description: str = Field(description="One-line human-readable description.")
    topology: Topology = Field(description="Articulated body topology.")
    num_dofs: int = Field(gt=0, description="Total degrees of freedom.")
    component_names: list[str] = Field(
        min_length=2,
        description=(
            "Ordered list of allowed component_name values, base to "
            "end-effector. Defines the constrained taxonomy."
        ),
    )
    visual_cues: dict[str, str] = Field(
        description=(
            "Per-component identification rules keyed by component_name. "
            "Must cover every entry in component_names."
        ),
    )
    tiebreakers: list[str] = Field(
        default_factory=list,
        description="Disambiguation rules for visually similar components.",
    )
    completeness_required: bool = Field(
        default=True,
        description="If true, instruct the model that every role must appear.",
    )
    self_check: bool = Field(
        default=True,
        description="If true, append a self-check step.",
    )
    notes: str | None = Field(
        default=None, description="Free-form notes (not sent to the LLM)."
    )

    @model_validator(mode="after")
    def _check_visual_cues_cover_components(self) -> RobotPromptEntry:
        duplicates = sorted(
            name for name, count in Counter(self.component_names).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"component_names contains duplicate entries: {duplicates}"
            )

        missing = [c for c in self.component_names if c not in self.visual_cues]
        if missing:
            raise ValueError(f"visual_cues missing entries for components: {missing}")
        return self
