# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional integrations for externally produced physics evidence."""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AuthoredMassProperties": "vomp",
    "VompApplyResult": "vomp",
    "VompIntegrationError": "vomp",
    "VompVoxelField": "vomp",
    "apply_vomp_mass_properties": "vomp",
    "load_vomp_voxel_field": "vomp",
    "AOUSD_DEFORMABLE_PROPOSAL_URL": "vomp_deformable",
    "AOUSD_DEFORMABLE_SCHEMA_COMMIT": "vomp_deformable",
    "DEFAULT_MAX_DEFORMABLE_VOXELS": "vomp_deformable",
    "MaterialReductionPolicy": "vomp_deformable",
    "NEWTON_1_4_AOUSD_BASELINE_COMMIT": "vomp_deformable",
    "NEWTON_DEFORMABLE_PROFILE": "vomp_deformable",
    "VompDeformableApplyResult": "vomp_deformable",
    "VompElasticReduction": "vomp_deformable",
    "VompTetMesh": "vomp_deformable",
    "apply_vomp_volume_deformable": "vomp_deformable",
    "build_vomp_tet_mesh": "vomp_deformable",
    "reduce_vomp_elastic_field": "vomp_deformable",
    "VompPipelineResult": "vomp_pipeline",
    "VompDeformablePipelineResult": "vomp_pipeline",
    "VompPreparedEvidence": "vomp_pipeline",
    "VompRenderConfig": "vomp_pipeline",
    "prepare_vomp_evidence": "vomp_pipeline",
    "run_vomp_mass_pipeline": "vomp_pipeline",
    "run_vomp_volume_deformable_pipeline": "vomp_pipeline",
    "DEFAULT_VOMP_ARTIFACT_SHA256": "vomp_defaults",
    "DEFAULT_VOMP_REVISION": "vomp_defaults",
    "ExternalVompRunner": "vomp_runtime",
    "VompRunner": "vomp_runtime",
    "VompRunRequest": "vomp_runtime",
    "VompRunResult": "vomp_runtime",
    "VompRuntimeConfig": "vomp_runtime",
}
_SUBMODULES = frozenset(
    {"vomp", "vomp_defaults", "vomp_deformable", "vomp_pipeline", "vomp_runtime"}
)

__all__ = [
    "DEFAULT_VOMP_ARTIFACT_SHA256",
    "DEFAULT_VOMP_REVISION",
    "AOUSD_DEFORMABLE_PROPOSAL_URL",
    "AOUSD_DEFORMABLE_SCHEMA_COMMIT",
    "AuthoredMassProperties",
    "DEFAULT_MAX_DEFORMABLE_VOXELS",
    "ExternalVompRunner",
    "MaterialReductionPolicy",
    "NEWTON_1_4_AOUSD_BASELINE_COMMIT",
    "VompApplyResult",
    "VompDeformableApplyResult",
    "VompDeformablePipelineResult",
    "VompElasticReduction",
    "VompIntegrationError",
    "VompPipelineResult",
    "VompPreparedEvidence",
    "VompRenderConfig",
    "VompRunRequest",
    "VompRunResult",
    "VompRunner",
    "VompRuntimeConfig",
    "VompTetMesh",
    "VompVoxelField",
    "NEWTON_DEFORMABLE_PROFILE",
    "apply_vomp_mass_properties",
    "apply_vomp_volume_deformable",
    "build_vomp_tet_mesh",
    "load_vomp_voxel_field",
    "prepare_vomp_evidence",
    "reduce_vomp_elastic_field",
    "run_vomp_mass_pipeline",
    "run_vomp_volume_deformable_pipeline",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        if name not in _SUBMODULES:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = import_module(f"{__name__}.{name}")
    else:
        value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_SUBMODULES})
