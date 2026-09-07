# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material-assignment workflow contracts."""

from .decisions import (
    MATERIAL_DECISION_PATCH_SCHEMA_VERSION,
    VISIBLE_CANDIDATE_PRIMS_SCHEMA_VERSION,
    DecisionContractError,
    MaterialDecisionValidation,
    validate_material_decision,
)
from .evidence import (
    MaterialCandidatePolicy,
    build_material_authoring_evidence,
)
from .finalizer import (
    MaterialDecisionNormalization,
    MaterialDecisionPolicyError,
    MaterialFinalizationPolicy,
    finalize_material_policy,
    normalize_material_decision_policy,
)
from .grounding import (
    candidate_grounding_evidence,
    sample_grounding_pixels,
    write_material_grounding_diagnostics,
)
from .manifest import (
    MaterialManifestEntry,
    ResolvedMaterialManifest,
    load_material_manifest,
)
from .policy import (
    MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP,
    PAINTED_OR_SATURATED_MATERIAL_TAGS,
    STRUCTURED_FINALIZER_GUARDRAILS,
    structured_finalizer_guardrail_prompt,
    structured_finalizer_rejection,
)
from .workflow import (
    MAX_MATERIAL_APPLY_ASSIGNMENT_GROUPS,
    MAX_MATERIAL_APPLY_TARGET_PATHS,
    MaterialGenerationRequest,
    MaterialGenerationResult,
    MaterialGoalDecision,
    MaterialSceneSession,
    MaterialVqaGoal,
    execute_material_generation,
    partition_material_assignments,
)

__all__ = [
    "MATERIAL_DECISION_PATCH_SCHEMA_VERSION",
    "VISIBLE_CANDIDATE_PRIMS_SCHEMA_VERSION",
    "MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP",
    "MAX_MATERIAL_APPLY_ASSIGNMENT_GROUPS",
    "MAX_MATERIAL_APPLY_TARGET_PATHS",
    "PAINTED_OR_SATURATED_MATERIAL_TAGS",
    "STRUCTURED_FINALIZER_GUARDRAILS",
    "DecisionContractError",
    "MaterialDecisionValidation",
    "MaterialCandidatePolicy",
    "MaterialDecisionPolicyError",
    "MaterialDecisionNormalization",
    "MaterialFinalizationPolicy",
    "MaterialGenerationRequest",
    "MaterialGenerationResult",
    "MaterialGoalDecision",
    "MaterialManifestEntry",
    "ResolvedMaterialManifest",
    "MaterialSceneSession",
    "MaterialVqaGoal",
    "load_material_manifest",
    "build_material_authoring_evidence",
    "candidate_grounding_evidence",
    "execute_material_generation",
    "finalize_material_policy",
    "normalize_material_decision_policy",
    "sample_grounding_pixels",
    "partition_material_assignments",
    "write_material_grounding_diagnostics",
    "structured_finalizer_guardrail_prompt",
    "structured_finalizer_rejection",
    "validate_material_decision",
]
