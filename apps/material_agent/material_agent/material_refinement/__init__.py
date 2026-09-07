# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendered material refinement and variation over shared optimization."""

from .contracts import (
    MaterialObjectiveWeights,
    MaterialRefinementConfig,
    MaterialRefinementGoal,
    MaterialRefinementJudgeSettings,
    MaterialRefinementRenderSettings,
    MaterialRefinementSearchSettings,
    MaterialRefinementSource,
    TextureVariationSettings,
)
from .objective import (
    MaterialFeatures,
    MaterialObjectiveScore,
    extract_material_features,
    material_feature_distance,
    resolve_target_features,
    score_material_features,
)
from .runner import (
    MaterialRefinementCancelled,
    MaterialRefinementDependencyError,
    MaterialRefinementError,
    MaterialRefinementResult,
    run_material_refinement,
)
from .texture_variation import (
    RestTextureVariationGenerator,
    TextureVariationArtifacts,
    TextureVariationError,
    TextureVariationGenerator,
    TextureVariationRequest,
)
from .variation import (
    MaterialVariationConfig,
    MaterialVariationGoal,
    MaterialVariationResult,
    MaterialVariationSlot,
    run_material_variations,
)

__all__ = [
    "MaterialFeatures",
    "MaterialObjectiveScore",
    "MaterialRefinementGoal",
    "MaterialRefinementDependencyError",
    "MaterialRefinementJudgeSettings",
    "MaterialRefinementCancelled",
    "MaterialRefinementConfig",
    "MaterialRefinementError",
    "MaterialRefinementResult",
    "MaterialRefinementRenderSettings",
    "MaterialRefinementSearchSettings",
    "MaterialRefinementSource",
    "MaterialObjectiveWeights",
    "MaterialVariationConfig",
    "MaterialVariationGoal",
    "MaterialVariationResult",
    "MaterialVariationSlot",
    "RestTextureVariationGenerator",
    "TextureVariationArtifacts",
    "TextureVariationError",
    "TextureVariationGenerator",
    "TextureVariationRequest",
    "TextureVariationSettings",
    "extract_material_features",
    "material_feature_distance",
    "resolve_target_features",
    "run_material_refinement",
    "run_material_variations",
    "score_material_features",
]
