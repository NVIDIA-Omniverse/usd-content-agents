# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene decomposition workflow contracts for agentic asset workflows."""

from .decomposition import decompose_scene, run_scene_decomposition
from .manifest import (
    ArtifactReference,
    DecomposedAsset,
    DecompositionPhaseResult,
    DecompositionPolicy,
    ManifestCatalog,
    ManifestCatalogEntry,
    SceneDecompositionManifest,
    SceneDecompositionRequest,
    SceneDecompositionResult,
    SceneInstanceGroup,
    ScenePayloadGroup,
    ScenePrototypeGroup,
    StageMetadata,
)

__all__ = [
    "ArtifactReference",
    "DecompositionPhaseResult",
    "DecomposedAsset",
    "DecompositionPolicy",
    "ManifestCatalog",
    "ManifestCatalogEntry",
    "SceneDecompositionManifest",
    "SceneDecompositionRequest",
    "SceneDecompositionResult",
    "SceneInstanceGroup",
    "ScenePayloadGroup",
    "ScenePrototypeGroup",
    "StageMetadata",
    "decompose_scene",
    "run_scene_decomposition",
]
