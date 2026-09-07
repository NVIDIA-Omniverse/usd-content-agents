# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical source-asset to SimReady workflow contracts."""

from .final_render import (
    CAD_TO_SIMREADY_FINAL_RENDER_SCHEMA_VERSION,
    render_final_simready_evidence,
)
from .workflow import (
    CAD_TO_SIMREADY_FLATTEN_SCHEMA_VERSION,
    CAD_TO_SIMREADY_PREFLIGHT_STEPS,
    CAD_TO_SIMREADY_REQUEST_SCHEMA_VERSION,
    CAD_TO_SIMREADY_RESULT_SCHEMA_VERSION,
    CAD_TO_SIMREADY_STAGE_ORDER,
    CAD_TO_SIMREADY_STEPS,
    CAD_TO_SIMREADY_WORKFLOW_ENTRYPOINT,
    CAD_TO_SIMREADY_WORKFLOW_SKILL,
    CadToSimReadyArtifactBinding,
    CadToSimReadyInvocation,
    CadToSimReadyPreflightRecord,
    CadToSimReadyPreflightStep,
    CadToSimReadyRequest,
    CadToSimReadyResult,
    CadToSimReadyStageRecord,
    CadToSimReadyStep,
    bind_artifact,
    build_cad_to_simready_result,
    build_preflight_record,
    build_stage_record,
    execute_cad_to_simready_workflow,
    flatten_usd_for_physics,
    source_format,
)

__all__ = [
    "CAD_TO_SIMREADY_FINAL_RENDER_SCHEMA_VERSION",
    "CAD_TO_SIMREADY_FLATTEN_SCHEMA_VERSION",
    "CAD_TO_SIMREADY_PREFLIGHT_STEPS",
    "CAD_TO_SIMREADY_REQUEST_SCHEMA_VERSION",
    "CAD_TO_SIMREADY_RESULT_SCHEMA_VERSION",
    "CAD_TO_SIMREADY_STAGE_ORDER",
    "CAD_TO_SIMREADY_STEPS",
    "CAD_TO_SIMREADY_WORKFLOW_ENTRYPOINT",
    "CAD_TO_SIMREADY_WORKFLOW_SKILL",
    "CadToSimReadyArtifactBinding",
    "CadToSimReadyInvocation",
    "CadToSimReadyPreflightRecord",
    "CadToSimReadyPreflightStep",
    "CadToSimReadyRequest",
    "CadToSimReadyResult",
    "CadToSimReadyStageRecord",
    "CadToSimReadyStep",
    "bind_artifact",
    "build_cad_to_simready_result",
    "build_preflight_record",
    "build_stage_record",
    "execute_cad_to_simready_workflow",
    "flatten_usd_for_physics",
    "render_final_simready_evidence",
    "source_format",
]
