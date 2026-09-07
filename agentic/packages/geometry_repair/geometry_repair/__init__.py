# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile-driven deterministic geometry repair."""

from .advanced_profiles import AdvancedProfileEvidenceReport, AdvancedProfileRequest
from .collision_audit import SourceCollisionAudit, audit_source_collision_proxy
from .correspondence import CorrespondenceEvidence, build_mesh_correspondence
from .diagnosis import diagnose_asset
from .evaluation import phase2_candidate_decisions, run_cgal_exact_audit
from .format_validation import validate_source_format
from .models import (
    ClassifiedHoleIntent,
    Diagnosis,
    DiagnosisIssue,
    FidelityReport,
    GeometryMetrics,
    GeometryRole,
    ProtectedFeature,
    ProtectedFeatureProbe,
    RepairBudgets,
    RepairCertificate,
    RepairIntent,
    RepairPlan,
    RepairRequest,
    RepairResult,
    RoleValidation,
    SourceFormatValidationReport,
)
from .orchestrator import run_geometry_repair
from .protected_features import (
    ProtectedFeatureCandidateReport,
    detect_protected_feature_candidates,
)
from .scalable_audit import ScalableAuditBudget, ScalableMeshAuditReport, audit_triangle_mesh
from .usd_articulation import (
    NestedRigidBodyNormalization,
    UsdArticulationExtraction,
    extract_usd_articulation_request,
    normalize_nested_rigid_body_xforms,
)
from .usd_intake import UsdIntakeReport, inventory_usd_stage

__all__ = [
    "ClassifiedHoleIntent",
    "Diagnosis",
    "DiagnosisIssue",
    "FidelityReport",
    "GeometryRole",
    "GeometryMetrics",
    "ProtectedFeature",
    "ProtectedFeatureProbe",
    "RepairBudgets",
    "RepairCertificate",
    "RepairIntent",
    "RepairPlan",
    "RepairRequest",
    "RepairResult",
    "RoleValidation",
    "SourceFormatValidationReport",
    "AdvancedProfileEvidenceReport",
    "AdvancedProfileRequest",
    "CorrespondenceEvidence",
    "ProtectedFeatureCandidateReport",
    "SourceCollisionAudit",
    "ScalableAuditBudget",
    "ScalableMeshAuditReport",
    "UsdIntakeReport",
    "UsdArticulationExtraction",
    "NestedRigidBodyNormalization",
    "audit_triangle_mesh",
    "audit_source_collision_proxy",
    "build_mesh_correspondence",
    "diagnose_asset",
    "detect_protected_feature_candidates",
    "phase2_candidate_decisions",
    "run_cgal_exact_audit",
    "run_geometry_repair",
    "validate_source_format",
    "inventory_usd_stage",
    "extract_usd_articulation_request",
    "normalize_nested_rigid_body_xforms",
]
