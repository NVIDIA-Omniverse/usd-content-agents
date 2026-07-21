# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated material library helpers for Material Agent."""

from material_agent.material_library_generation.builder import (
    build_generated_material_library,
)
from material_agent.material_library_generation.conditioning import (
    MATERIAL_CONDITIONING_MANIFEST_NAME,
    MATERIAL_CONDITIONING_SCHEMA_VERSION,
    OVRTX_CONDITIONING_SCHEMA_VERSION,
    REAL_SEED_MATERIAL_SCHEMA_VERSION,
    MaterialConditioningEvidenceMode,
    MaterialConditioningOptions,
    MaterialConditioningResult,
    RealMaterialConditioningInputs,
    prepare_material_conditioning,
)
from material_agent.material_library_generation.creation import (
    MaterialCreationBackendRegistry,
    create_material_package,
)
from material_agent.material_library_generation.creation_contract import (
    MATERIAL_CREATION_MANIFEST_NAME,
    MATERIAL_CREATION_SCHEMA_VERSION,
    MATERIAL_LIBRARY_NAME,
    MATERIAL_LIST_MANIFEST_NAME,
    BackendMaterialResult,
    CreatedMaterial,
    CreatedMaterialListEntry,
    CreateMaterialRequest,
    MaterialAction,
    MaterialArtifactLayout,
    MaterialChannel,
    MaterialChannelArtifact,
    MaterialChannelComponent,
    MaterialChannelSource,
    MaterialColorSpace,
    MaterialComponentProvenance,
    MaterialConditioningArtifact,
    MaterialConditioningArtifactSource,
    MaterialConditioningKind,
    MaterialCreationBackend,
    MaterialCreationDiagnostic,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationMode,
    MaterialCreationProvenance,
    MaterialDegradation,
    MaterialDegradationCode,
    MaterialDiagnosticSeverity,
    NormalConvention,
    ORMPacking,
    PreparedMaterialConditioning,
    intended_part_prim_path_hints,
)
from material_agent.material_library_generation.fake_backend import (
    FakeMaterialBackendBehavior,
    FakeMaterialCreationBackend,
)
from material_agent.material_library_generation.prototypes import (
    MaterialPrototype,
    load_material_prototypes_from_data,
    load_material_prototypes_from_manifest,
    score_material_prototype,
    select_material_prototype,
)
from material_agent.material_library_generation.schema import (
    DEFAULT_LIBRARY_ROOT,
    GeneratedMaterial,
    GeneratedMaterialLibrary,
    IntendedPart,
    MaterialGenerationPlan,
    MaterialRecipe,
    PBRHints,
    TextureMapSet,
    make_material_id,
    make_usd_identifier,
)
from material_agent.material_library_generation.texture_generation import (
    TextureGenerationSettings,
    generate_texture_maps,
)
from material_agent.material_library_generation.texture_space_diffusion_backend import (
    TEXTURE_SPACE_DIFFUSION_ADAPTER_REVISION,
    TEXTURE_SPACE_DIFFUSION_BACKEND_NAME,
    TEXTURE_SPACE_DIFFUSION_UNAVAILABLE_REVISION,
    TextureSpaceDiffusionBackendConfig,
    TextureSpaceDiffusionMaterialCreationBackend,
)
from material_agent.material_library_generation.validation import (
    ValidationResult,
    validate_generated_material_library,
)
from material_agent.material_library_generation.videomatgen_backend import (
    VIDEOMATGEN_ADAPTER_REVISION,
    VIDEOMATGEN_BACKEND_NAME,
    VIDEOMATGEN_UNAVAILABLE_REVISION,
    VideoMatGenBackendConfig,
    VideoMatGenMaterialCreationBackend,
)

__all__ = [
    "DEFAULT_LIBRARY_ROOT",
    "BackendMaterialResult",
    "CreateMaterialRequest",
    "CreatedMaterial",
    "CreatedMaterialListEntry",
    "FakeMaterialBackendBehavior",
    "FakeMaterialCreationBackend",
    "GeneratedMaterial",
    "GeneratedMaterialLibrary",
    "IntendedPart",
    "MATERIAL_CREATION_MANIFEST_NAME",
    "MATERIAL_CREATION_SCHEMA_VERSION",
    "MATERIAL_CONDITIONING_MANIFEST_NAME",
    "MATERIAL_CONDITIONING_SCHEMA_VERSION",
    "OVRTX_CONDITIONING_SCHEMA_VERSION",
    "REAL_SEED_MATERIAL_SCHEMA_VERSION",
    "MATERIAL_LIBRARY_NAME",
    "MATERIAL_LIST_MANIFEST_NAME",
    "MaterialAction",
    "MaterialArtifactLayout",
    "MaterialCreationBackendRegistry",
    "MaterialChannel",
    "MaterialChannelArtifact",
    "MaterialChannelComponent",
    "MaterialChannelSource",
    "MaterialColorSpace",
    "MaterialComponentProvenance",
    "MaterialConditioningEvidenceMode",
    "MaterialConditioningOptions",
    "MaterialConditioningResult",
    "MaterialConditioningArtifact",
    "MaterialConditioningArtifactSource",
    "MaterialConditioningKind",
    "MaterialCreationBackend",
    "MaterialCreationDiagnostic",
    "MaterialCreationError",
    "MaterialCreationErrorCode",
    "MaterialCreationMode",
    "MaterialCreationProvenance",
    "MaterialDegradation",
    "MaterialDegradationCode",
    "MaterialDiagnosticSeverity",
    "MaterialPrototype",
    "MaterialGenerationPlan",
    "MaterialRecipe",
    "NormalConvention",
    "ORMPacking",
    "PBRHints",
    "PreparedMaterialConditioning",
    "RealMaterialConditioningInputs",
    "TextureGenerationSettings",
    "TextureSpaceDiffusionBackendConfig",
    "TextureSpaceDiffusionMaterialCreationBackend",
    "TextureMapSet",
    "TEXTURE_SPACE_DIFFUSION_ADAPTER_REVISION",
    "TEXTURE_SPACE_DIFFUSION_BACKEND_NAME",
    "TEXTURE_SPACE_DIFFUSION_UNAVAILABLE_REVISION",
    "VIDEOMATGEN_ADAPTER_REVISION",
    "VIDEOMATGEN_BACKEND_NAME",
    "VIDEOMATGEN_UNAVAILABLE_REVISION",
    "ValidationResult",
    "VideoMatGenBackendConfig",
    "VideoMatGenMaterialCreationBackend",
    "build_generated_material_library",
    "create_material_package",
    "generate_texture_maps",
    "intended_part_prim_path_hints",
    "load_material_prototypes_from_data",
    "load_material_prototypes_from_manifest",
    "make_material_id",
    "make_usd_identifier",
    "prepare_material_conditioning",
    "score_material_prototype",
    "select_material_prototype",
    "validate_generated_material_library",
]

_LAZY_STEP1X_EXPORTS = {
    "Step1XMaterialCreationBackend",
    "Step1XMaterialCreationConfig",
}


def __getattr__(name: str) -> object:
    """Load optional Step1X backend exports only when callers ask for them."""

    if name in _LAZY_STEP1X_EXPORTS:
        from material_agent.material_library_generation.step1x_backend import (
            Step1XMaterialCreationBackend,
            Step1XMaterialCreationConfig,
        )

        exports = {
            "Step1XMaterialCreationBackend": Step1XMaterialCreationBackend,
            "Step1XMaterialCreationConfig": Step1XMaterialCreationConfig,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
