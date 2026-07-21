# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material-assignment workflow policy shared by wrappers and skills."""

from __future__ import annotations

PAINTED_OR_SATURATED_MATERIAL_TAGS = frozenset(
    {"paint", "orange", "yellow", "red", "blue", "white", "plastic"}
)
MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP = 16

STRUCTURED_FINALIZER_GUARDRAILS = (
    {
        "id": "slender_bar_metal_family",
        "prompt": (
            "`slender_bar` groups should remain in the metal family. Do not apply "
            "saturated paint, white plastic, or body-color materials to rollers, "
            "rods, rails, or bars."
        ),
        "rejection": (
            "Rejected saturated/painted material on slender-bar geometry; rollers, "
            "rods, rails, and bars must remain in the metal family."
        ),
    },
    {
        "id": "split_broad_painted_mixed_groups",
        "prompt": (
            "Painted or saturated assignments spanning more than three prims must "
            "not mix unrelated shape hints. Split the decision into clearly visible "
            "panel/body paths and leave unrelated paths preserved or ambiguous."
        ),
        "rejection": (
            "Rejected broad painted assignment on a mixed-shape group; split the "
            "decision into clearly visible panel/body paths and leave unrelated "
            "paths preserved or ambiguous."
        ),
    },
    {
        "id": "split_large_mixed_groups",
        "prompt": (
            f"Material assignment groups above {MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP} prims "
            "must have a single shape hint. Split large mixed groups by material "
            "family or shape."
        ),
        "rejection": (
            "Rejected broad material assignment on a large mixed group; use a more "
            "specific material family or split by shape."
        ),
    },
)


def structured_finalizer_guardrail_prompt() -> str:
    """Return prompt text for guardrails enforced during finalization."""

    return "\n".join(
        f"- [{rule['id']}] {rule['prompt']}" for rule in STRUCTURED_FINALIZER_GUARDRAILS
    )


def structured_finalizer_rejection(rule_id: str) -> str:
    """Return the rejection message for a structured finalizer guardrail."""

    for rule in STRUCTURED_FINALIZER_GUARDRAILS:
        if rule["id"] == rule_id:
            return f"{rule['rejection']} (guardrail: {rule_id})"
    raise ValueError(f"Unknown structured finalizer guardrail: {rule_id}")
