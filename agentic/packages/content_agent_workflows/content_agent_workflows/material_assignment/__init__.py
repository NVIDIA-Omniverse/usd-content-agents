# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material-assignment workflow contracts."""

from .policy import (
    MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP,
    PAINTED_OR_SATURATED_MATERIAL_TAGS,
    STRUCTURED_FINALIZER_GUARDRAILS,
    structured_finalizer_guardrail_prompt,
    structured_finalizer_rejection,
)

__all__ = [
    "MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP",
    "PAINTED_OR_SATURATED_MATERIAL_TAGS",
    "STRUCTURED_FINALIZER_GUARDRAILS",
    "structured_finalizer_guardrail_prompt",
    "structured_finalizer_rejection",
]
