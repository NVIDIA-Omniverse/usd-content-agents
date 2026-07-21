# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent Functions."""

from joint_agent.functions.analytic_gprim_normalization import (
    ANALYTIC_GPRIM_COLLIDER_CHILD_NAME,
    ANALYTIC_GPRIM_NORMALIZATION_VERSION,
    ANALYTIC_GPRIM_RECEIPT_SCHEMA_VERSION,
    ANALYTIC_GPRIM_RENDER_CHILD_NAME,
    AnalyticGprimNormalizationError,
    AnalyticGprimNormalizationResult,
    normalize_analytic_cube_gprims,
)
from joint_agent.functions.articulation_contract import (
    ARTICULATION_CONTRACT_SCHEMA_VERSION,
    ArticulationContractV1,
    ArticulationRecordV1,
    BodyAuthoring,
    ContractDiagnosticV1,
    ContractSummaryV1,
    DiagnosticSeverity,
    JointRecordV1,
    LinkRecordV1,
    LinkRole,
    PrimRecordV1,
)
from joint_agent.functions.articulation_contract_stage2 import (
    build_articulation_contract_from_stage2,
)
from joint_agent.functions.articulation_types import ArticulationReviewStatus
from joint_agent.functions.consistency import apply_prediction_consistency
from joint_agent.functions.inference import batch_classify_assets, classify_asset
from joint_agent.functions.joint_rigger_contract_bridge import (
    build_joint_rigger_input_from_contract,
)
from joint_agent.functions.joint_rigger_gate3_plan import (
    GATE3_PHYSICS_PLAN_ENVELOPE_SCHEMA_VERSION,
    Gate3PhysicsPlanEnvelopeV1,
    build_gate3_joint_rigger_input_from_contract,
)
from joint_agent.functions.joint_rigger_gate3_policy import (
    GATE3_PHYSICS_POLICY_SCHEMA_VERSION,
    SUPPORTED_COLLIDER_PRIM_TYPES,
    ColliderPrimType,
    Gate3PhysicsPolicyV1,
    produce_owner_approved_gate3_joint_rigger_input,
)
from joint_agent.functions.physics_schema_authoring import author_physics_schemas
from joint_agent.functions.rigged_reference_validation import (
    compare_articulation_candidates_to_reference,
    extract_reference_articulation_manifest,
    write_rigged_reference_validation_report_html,
)
from joint_agent.functions.stage1_schema import (
    Stage1PredictionContract,
    has_parseable_stage1_source_response,
    normalize_stage1_prediction_payload,
    stage1_prediction_json_schema,
    unwrap_stage1_prediction_payload,
    validate_stage1_prediction_payload,
)

__all__ = [
    "ANALYTIC_GPRIM_COLLIDER_CHILD_NAME",
    "ANALYTIC_GPRIM_NORMALIZATION_VERSION",
    "ANALYTIC_GPRIM_RECEIPT_SCHEMA_VERSION",
    "ANALYTIC_GPRIM_RENDER_CHILD_NAME",
    "ARTICULATION_CONTRACT_SCHEMA_VERSION",
    "AnalyticGprimNormalizationError",
    "AnalyticGprimNormalizationResult",
    "GATE3_PHYSICS_POLICY_SCHEMA_VERSION",
    "GATE3_PHYSICS_PLAN_ENVELOPE_SCHEMA_VERSION",
    "SUPPORTED_COLLIDER_PRIM_TYPES",
    "ArticulationContractV1",
    "ArticulationRecordV1",
    "ArticulationReviewStatus",
    "BodyAuthoring",
    "ContractDiagnosticV1",
    "ContractSummaryV1",
    "ColliderPrimType",
    "DiagnosticSeverity",
    "Gate3PhysicsPolicyV1",
    "Gate3PhysicsPlanEnvelopeV1",
    "JointRecordV1",
    "LinkRole",
    "LinkRecordV1",
    "PrimRecordV1",
    "Stage1PredictionContract",
    "apply_prediction_consistency",
    "classify_asset",
    "batch_classify_assets",
    "has_parseable_stage1_source_response",
    "build_joint_rigger_input_from_contract",
    "author_physics_schemas",
    "build_articulation_contract_from_stage2",
    "build_gate3_joint_rigger_input_from_contract",
    "compare_articulation_candidates_to_reference",
    "extract_reference_articulation_manifest",
    "normalize_analytic_cube_gprims",
    "normalize_stage1_prediction_payload",
    "produce_owner_approved_gate3_joint_rigger_input",
    "stage1_prediction_json_schema",
    "unwrap_stage1_prediction_payload",
    "validate_stage1_prediction_payload",
    "write_rigged_reference_validation_report_html",
]
