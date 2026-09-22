# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic geometry/CAD workflow implementation."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Literal, cast

from cad_verifier.sim_ready import (
    OFFICIAL_SIMREADY_DEFAULT_VERSION,
    STATIC_VISUAL_PROFILE_ID,
    resolve_official_simready_profile,
)
from geometry_repair import (
    ClassifiedHoleIntent,
    ProtectedFeature,
    RepairBudgets,
    RepairIntent,
    RepairRequest,
    run_geometry_repair,
)
from geometry_repair import (
    RepairResult as GeometryRepairResult,
)
from geometry_repair.models import RepairProfile
from geometry_repair.worker_ids import canonical_worker_names
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.validation_evidence import (
    EvidenceArtifact,
    ValidationCheck,
    ValidationEvidence,
)
from content_agent_workflows.geometry.policy import (
    GeometryRouteDecision,
    route_geometry_request,
)
from content_agent_workflows.runtime_validation import (
    RuntimeValidationMode,
    RuntimeValidationRequest,
    run_runtime_validation,
)
from content_agent_workflows.simready.conform_profile import (
    run_simready_profile_conformance,
)
from content_agent_workflows.simready.models import (
    SimReadyConformanceInput,
    SimReadyValidationInput,
)
from content_agent_workflows.simready.validate_profile import (
    run_simready_profile_validation,
)

from . import scene_ops
from .audit import (
    GeometryAssetAuditReport,
    audit_geometry_asset,
    write_audit_artifacts,
)
from .evidence import (
    GeometryEvidenceBundle,
    evidence_reference,
    write_geometry_evidence_bundle,
)
from .rendering import (
    GeometryRenderCandidateBinding,
    GeometryRenderPreset,
    OvRTXRenderMode,
    render_geometry_evidence,
    verify_geometry_render_candidate_evidence,
)
from .segmentation import (
    GeometrySegmentationHandoff,
    SegmentationOutcome,
    consume_segmentation_handoff,
    segmentation_handoff_from_routing,
    segmentation_validation_check,
)
from .segmentation_routing import (
    GeometrySegmentationRoutingDecision,
    SegmentationRoute,
    route_segmentation,
)
from .source_prep import (
    BREP_SUFFIXES,
    PreparedGeometrySource,
    SourceAuthoringMode,
    admit_geometry_source_bundle,
    prepare_geometry_source,
    verify_geometry_source_identity,
)
from .validation import run_geometry_usd_validation

GEOMETRY_HANDOFF_MANIFEST_SCHEMA_VERSION = "content-agent-workflows.geometry-handoff.v3"
GEOMETRY_WORKFLOW_SCHEMA_VERSION = "content-agent-workflows.geometry.v3"
GeometryRuntimeEngine = Literal["ovphysx", "fake", "none"]
GeometryOptimizationPolicy = scene_ops.GeometryOptimizationPolicy
GeometrySimReadyMode = Literal["skip", "validate", "validate_and_route_conformance"]
GeometryRepairMode = Literal["off", "diagnose", "auto"]
GeometryHandoffReady = Literal["yes", "conditional", "no"]
_NATIVE_BREP_FORMATS = frozenset({"brep", "iges", "igs", "step", "stp"})


class GeometryWorkflowInput(BaseModel):
    """Input for the agentic geometry workflow."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    source_path: Path | None = None
    source_manifest_path: Path | None = None
    expected_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_source_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_representation_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    source_representation_role: str | None = None
    prompt: str | None = None
    image_path: Path | None = None
    generated_usd_path: Path | None = None
    output_dir: Path
    output_usd_path: Path | None = None
    source_authoring_mode: SourceAuthoringMode = "auto"
    render_topology_policy: Literal["strict", "preserve_source"] = "strict"
    allow_lossy_recovery: bool = False
    target_profile: str = STATIC_VISUAL_PROFILE_ID
    target_runtime: str = "isaac-lab"
    run_runtime_validation: bool | None = None
    runtime_validation_mode: RuntimeValidationMode = "skip"
    runtime_engine: GeometryRuntimeEngine = "ovphysx"
    runtime_duration_s: float = 3.0
    runtime_dt: float = 1.0 / 120.0
    runtime_sample_fps: int = 30
    render_evidence: bool = False
    render_preset: GeometryRenderPreset = "six_view"
    render_backend: Literal["ovrtx", "remote"] = "ovrtx"
    render_remote_base_url: str | None = None
    render_remote_api_key: SecretStr | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    render_remote_allow_unauthenticated_identity: bool = False
    render_ovrtx_mode: OvRTXRenderMode | None = "pt"
    render_ovrtx_num_sensor_updates: int | None = Field(default=64, ge=1)
    optimization_policy: GeometryOptimizationPolicy = "preserve_correspondence"
    optimizer_backend: Literal["local", "remote"] = "local"
    optimization_config: dict[str, Any] = Field(default_factory=dict)
    canonicalize_stage_metrics: bool = False
    segmentation_run_dir: Path | None = None
    segmentation_required: bool = False
    segmentation_required_parts: list[str] = Field(default_factory=list)
    segmentation_require_ovrtx_evidence: bool = True
    repair_mode: GeometryRepairMode = "off"
    repair_profile: RepairProfile = "visual_only"
    repair_profile_confirmed: bool = True
    repair_production_use: bool = True
    repair_protected_features: list[ProtectedFeature] = Field(default_factory=list)
    repair_classified_holes: list[ClassifiedHoleIntent] = Field(default_factory=list)
    repair_proposed_intents: list[RepairIntent] = Field(default_factory=list)
    repair_use_proposed_intent_ranking: bool = False
    repair_budgets: RepairBudgets = Field(default_factory=RepairBudgets)
    repair_enabled_workers: list[str] | None = None
    repair_deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    repair_collision_runtime_engine: Literal["skip", "fake", "ovphysx"] = "skip"
    repair_source_uri: str | None = None
    repair_source_license: str | None = None
    repair_source_provenance: dict[str, Any] = Field(default_factory=dict)
    repair_dependency_roots: list[Path] = Field(default_factory=list)
    repair_dependency_remap_manifest: dict[str, Any] | None = None
    repair_advanced_profile: dict[str, Any] | None = None
    install_missing_converters: bool = False
    converter_timeout_s: float = Field(default=120.0, gt=0.0)
    run_shared_usd_validation: bool = True
    run_legacy_cad_physics_preflight: bool = False
    simready_mode: GeometrySimReadyMode = "skip"
    simready_profile: str | None = None
    simready_profile_version: str = OFFICIAL_SIMREADY_DEFAULT_VERSION
    simready_install_missing: bool = False
    fail_on_validation_error: bool = False
    run_asset_audit: bool = True
    audit_context: dict[str, Any] = Field(default_factory=dict)

    variant_id: str | None = None
    param_overrides: dict[str, Any] = Field(default_factory=dict)
    usd_tessellation_tolerance: float = Field(
        default=0.5,
        gt=0.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_preserved_render_policy(self):
        if self.render_topology_policy == "preserve_source" and (
            self.source_authoring_mode != "lossless_gltf"
            or self.optimization_policy != "skip"
            or self.repair_mode != "off"
            or self.canonicalize_stage_metrics
            or self.run_legacy_cad_physics_preflight
        ):
            raise ValueError("preserve_source render topology requires explicit lossless_gltf intake, skipped optimization, no render repair/metric rewrite, and delegated downstream physics")
        return self

    @field_validator("repair_enabled_workers", mode="after")
    @classmethod
    def _canonicalize_repair_workers(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return cast(list[str], canonical_worker_names(value))

    @field_validator("output_usd_path", mode="after")
    @classmethod
    def _require_supported_output_usd_path(cls, value: Path | None) -> Path | None:
        if value is not None and value.suffix.lower() not in {
            ".usd",
            ".usda",
            ".usdc",
        }:
            raise ValueError("output_usd_path must end in .usd, .usda, or .usdc")
        return value

    @model_validator(mode="after")
    def _map_legacy_runtime_flag(self) -> GeometryWorkflowInput:
        if self.run_runtime_validation is not None:
            mapped_mode: RuntimeValidationMode = (
                "temporary_loadability_proxy" if self.run_runtime_validation else "skip"
            )
            if "runtime_validation_mode" not in self.model_fields_set:
                self.runtime_validation_mode = mapped_mode
        if self.repair_advanced_profile is not None:
            if self.repair_profile not in {"articulated_rigid", "contact_rich"}:
                raise ValueError(
                    "repair_advanced_profile requires articulated_rigid or contact_rich"
                )
            if self.repair_advanced_profile.get("profile") != self.repair_profile:
                raise ValueError(
                    "repair_advanced_profile.profile must match repair_profile"
                )
        if len(self.segmentation_required_parts) != len(
            set(self.segmentation_required_parts)
        ):
            raise ValueError("segmentation_required_parts must not contain duplicates")
        for name in self.segmentation_required_parts:
            if (
                not name
                or len(name) > 2048
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or any(ord(character) < 32 for character in name)
            ):
                raise ValueError(
                    "segmentation_required_parts must contain bounded plain names"
                )
        if self.segmentation_required_parts:
            self.segmentation_required = True
        return self


class GeometryWorkflowResult(BaseModel):
    """Result and canonical artifacts from a geometry workflow run."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    route: GeometryRouteDecision
    output_dir: str
    source_usd_path: str | None = None
    prepared_source_path: str | None = None
    source_bundle_manifest_path: str | None = None
    source_bundle_id: str | None = None
    source_revision: str | None = None
    source_provider_id: str | None = None
    source_representation_id: str | None = None
    representation_artifacts: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict
    )
    source_fidelity_tier: str | None = None
    conversion_probe_path: str | None = None
    conversion_report_path: str | None = None
    optimized_usd_path: str | None = None
    geometry_usd_path: str | None = None
    inspection_path: str | None = None
    optimization_metadata_path: str | None = None
    mesh_normalization_path: str | None = None
    segmentation_outcome: SegmentationOutcome = "not_requested"
    segmentation_route: SegmentationRoute | None = None
    segmentation_run_dir: str | None = None
    segmentation_routing_path: str | None = None
    segmentation_manifest_path: str | None = None
    segmentation_usd_path: str | None = None
    segmentation_validation_path: str | None = None
    handoff_manifest_path: str | None = None
    validation_evidence_path: str | None = None
    runtime_report_path: str | None = None
    render_report_path: str | None = None
    usd_validation_report_path: str | None = None
    simready_report_path: str | None = None
    simready_conformance_report_path: str | None = None
    evidence_bundle_path: str | None = None
    authoring_contract_path: str | None = None
    render_audit_path: str | None = None
    asset_audit_path: str | None = None
    asset_audit_summary_path: str | None = None
    repair_outcome: str | None = None
    repair_diagnosis_path: str | None = None
    repair_plan_path: str | None = None
    repair_certificate_path: str | None = None
    repair_manifest_path: str | None = None
    repair_collision_usd_path: str | None = None
    repair_usd_intake_path: str | None = None
    repair_source_format_validation_path: str | None = None
    repair_dependency_localization_path: str | None = None
    repair_scalable_audit_path: str | None = None
    repair_correspondence_path: str | None = None
    repair_source_collision_audit_path: str | None = None
    repair_protected_feature_candidates_path: str | None = None
    repair_manifold_seam_analysis_path: str | None = None
    repair_advanced_profile_evidence_path: str | None = None
    native_repair_certificate_path: str | None = None
    variant_id: str | None = None
    param_overrides: dict[str, Any] = Field(default_factory=dict)
    validation_status: str = "not_evaluated"
    handoff_ready: GeometryHandoffReady = "no"
    error: str | None = None
    error_type: str | None = None
    error_traceback_path: str | None = None


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _as_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _variant_audit_requested(
    params: GeometryWorkflowInput,
    audit_context: dict[str, Any],
) -> bool:
    return bool(
        params.variant_id
        or params.param_overrides
        or audit_context.get("variant_id")
        or audit_context.get("param_overrides")
        or audit_context.get("variant_parameters")
        or audit_context.get("variant_rows")
        or audit_context.get("variant_proof")
    )


def _validation_status(
    checks: list[ValidationCheck], failures: list[str], warnings: list[str]
) -> str:
    if failures or any(check.status == "fail" for check in checks):
        return "fail"
    if warnings or any(check.status == "warning" for check in checks):
        return "conditional"
    return "pass"


def _handoff_ready(status: str) -> GeometryHandoffReady:
    if status == "pass":
        return "yes"
    if status == "conditional":
        return "conditional"
    return "no"


def _semantic_parts_for_handoff(
    metadata: dict[str, Any],
    segmentation: GeometrySegmentationHandoff,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add_record(record: dict[str, Any]) -> None:
        part_id = record.get("part_id")
        name = record.get("name")
        existing = next(
            (
                item
                for item in records
                if (
                    isinstance(part_id, str)
                    and part_id
                    and item.get("part_id") == part_id
                )
            ),
            None,
        )
        if (
            existing is None
            and not (isinstance(part_id, str) and part_id)
            and isinstance(name, str)
            and name
        ):
            named = [item for item in records if item.get("name") == name]
            if len(named) == 1:
                existing = named[0]
        if existing is None:
            records.append(record)
            return
        for key, value in record.items():
            if (
                key not in existing
                or existing[key] is None
                or existing[key] == ""
                or existing[key] == []
            ):
                existing[key] = value
            elif key == "role" and existing[key] != value:
                existing.setdefault("semantic_role", value)

    source_bundle = metadata.get("source_bundle")
    source_bundle_parts = (
        source_bundle.get("parts") if isinstance(source_bundle, dict) else None
    )
    if isinstance(source_bundle_parts, list):
        for source_part in source_bundle_parts:
            if not isinstance(source_part, dict):
                continue
            add_record(
                {
                    "name": source_part.get("name"),
                    "role": "provider_part",
                    "source": "geometry.source.v1",
                    "part_id": source_part.get("part_id"),
                    "parent_part_id": source_part.get("parent_part_id"),
                    "representation_ids": source_part.get("representation_ids") or [],
                    "transform": source_part.get("transform") or [],
                }
            )
    semantic_parts = metadata.get("semantic_parts")
    semantic_part_items = (
        semantic_parts if isinstance(semantic_parts, list | tuple) else ()
    )
    for semantic_part in semantic_part_items:
        if isinstance(semantic_part, dict):
            add_record(dict(semantic_part))
        elif isinstance(semantic_part, str) and semantic_part.strip():
            add_record(
                {
                    "name": semantic_part.strip(),
                    "role": "design_body",
                    "source": "provider_semantic_parts",
                }
            )
    by_name = {
        str(record.get("name")): record
        for record in records
        if isinstance(record.get("name"), str)
    }
    if (
        segmentation.routing is not None
        and segmentation.routing.route == "reuse_source_identity"
    ):
        for evidence in segmentation.routing.semantic_name_evidence:
            evidence_record: dict[str, Any] = {
                "name": evidence.name,
                "role": evidence.name,
                "source": "source_authored_identity",
                "identity_sources": list(evidence.sources),
                "source_prim_paths": list(evidence.prim_paths),
                "source_fidelity_tier": "source_asserted_identity",
                "validation_state": segmentation.outcome,
            }
            if len(evidence.prim_paths) == 1:
                evidence_record["output_prim_path"] = evidence.prim_paths[0]
            existing = by_name.get(evidence.name)
            if existing is None:
                records.append(evidence_record)
                by_name[evidence.name] = evidence_record
            else:
                existing["segmentation"] = evidence_record
    for part in segmentation.parts:
        part_record = part.manifest_record(
            source_asset_sha256=segmentation.source_asset_sha256,
            outcome=segmentation.outcome,
        )
        existing = by_name.get(part.name)
        if existing is None:
            records.append(part_record)
            by_name[part.name] = part_record
        else:
            existing["segmentation"] = part_record
            existing["output_prim_path"] = part.output_prim_path
            existing["source_fidelity_tier"] = part_record["source_fidelity_tier"]
    return records


def _cad_issue_message(item: Any, *, fallback: str) -> str:
    if isinstance(item, dict):
        return str(item.get("message") or item.get("code") or fallback)
    return str(item)


def _cad_preflight_checks(
    usd_path: Path,
    output_dir: Path,
    profile_id: str,
    *,
    include_legacy_physics: bool = False,
    preserved_render_reference: Path | None = None,
) -> tuple[
    list[ValidationCheck],
    list[str],
    list[str],
    list[EvidenceArtifact],
    Path,
]:
    checks: list[ValidationCheck] = []
    failures: list[str] = []
    warnings: list[str] = []
    reports: dict[str, dict[str, Any]] = {}
    required_levels: set[str] = {"mesh_topology"}
    report_functions: dict[str, Any] = {}
    try:
        from cad_verifier.sim_ready import (
            SIM_READY_PROFILES,
            audit_usd_mesh_topology,
            audit_usd_physics_authoring,
            audit_usd_physics_materials,
        )

        profile = SIM_READY_PROFILES.get(profile_id)
        required_levels = (
            set(profile.required_levels)
            if profile
            else {
                "mesh_topology",
                "physics_authoring",
                "physics_materials",
            }
        )
        report_functions = {
            "mesh_topology": audit_usd_mesh_topology,
            "physics_authoring": audit_usd_physics_authoring,
            "physics_materials": audit_usd_physics_materials,
        }
    except Exception as exc:
        reports["mesh_topology"] = {
            "status": "unavailable",
            "issues": [f"CAD verifier mesh preflight unavailable: {exc}"],
        }

    selected_levels = ["mesh_topology"]
    physics_levels = [
        level
        for level in ("physics_authoring", "physics_materials")
        if level in required_levels
    ]
    if include_legacy_physics:
        selected_levels.extend(physics_levels)
    for level in selected_levels:
        if level in reports:
            continue
        report_function = report_functions.get(level)
        if report_function is None:
            reports[level] = {
                "status": "unavailable",
                "issues": [f"CAD verifier {level} preflight is unavailable."],
            }
            continue
        try:
            reports[level] = dict(report_function(usd_path))
        except Exception as exc:
            reports[level] = {
                "status": "unavailable",
                "issues": [f"CAD verifier {level} preflight unavailable: {exc}"],
            }

    if preserved_render_reference is not None:
        from .lossless_gltf import verify_preserved_render

        fidelity = verify_preserved_render(preserved_render_reference, usd_path)
        raw_report = reports["mesh_topology"]
        # This is an explicitly scoped render handoff. Preserve the complete
        # strict report; it remains unsuitable evidence for collision cooking.
        if raw_report.get("status") in {"pass", "fail"}:
            reports["mesh_topology"] = {
                "status": "pass",
                "message": "Source render geometry is preserved, finite and well indexed; original visual topology is retained as diagnostics. Collision cooking/contact/runtime validation remains delegated and required before physical acceptance.",
                "issues": [],
                "metrics": raw_report.get("metrics", {}),
                "strict_topology_report": raw_report,
                "source_fidelity": fidelity,
                "claim_scope": "source_preserving_render_only",
                "physical_acceptance": False,
            }

    delegated_levels = physics_levels if not include_legacy_physics else []
    report_statuses = {
        str(report.get("status") or "not_evaluated") for report in reports.values()
    }
    known_report_statuses = {
        "pass",
        "fail",
        "error",
        "warn",
        "warning",
        "incomplete",
        "unavailable",
        "not_evaluated",
    }
    if report_statuses.intersection({"fail", "error"}):
        aggregate_status = "fail"
    elif report_statuses.difference(
        known_report_statuses
    ) or report_statuses.intersection({"warn", "warning", "incomplete", "unavailable"}):
        aggregate_status = "warning"
    elif report_statuses == {"not_evaluated"}:
        aggregate_status = "not_evaluated"
    elif "not_evaluated" in report_statuses:
        aggregate_status = "warning"
    else:
        aggregate_status = "pass"
    report_path = _write_json(
        output_dir / "cad_preflight_report.json",
        {
            "schema_version": "content-agent-workflows.geometry-cad-preflight.v2",
            "producer": "cad_verifier.sim_ready",
            "claim_scope": "cad_preflight",
            "target_profile_intent": profile_id,
            "status": aggregate_status,
            "legacy_physics_preflight_enabled": include_legacy_physics,
            "delegated_levels": delegated_levels,
            "reports": reports,
            "report": reports["mesh_topology"],
        },
    )
    artifacts = [
        EvidenceArtifact(
            kind="cad_preflight_report",
            path=str(report_path),
            description="CAD-specific mesh and optional legacy physics preflight evidence.",
            metadata={
                "producer": "cad_verifier.sim_ready",
                "claim_scope": "cad_preflight",
                "status": aggregate_status,
            },
        )
    ]

    for level in selected_levels:
        report = reports[level]
        status_value = str(report.get("status") or "not_evaluated")
        issues = [
            _cad_issue_message(item, fallback=f"CAD {level} preflight issue")
            for item in report.get("issues") or []
        ]
        if status_value in {"fail", "error"}:
            status: Literal["pass", "fail", "warning", "not_evaluated"] = "fail"
            check_failures = issues or [f"CAD {level} preflight failed."]
            check_warnings: list[str] = []
            failures.extend(check_failures)
        elif status_value in {"warn", "warning", "incomplete", "unavailable"}:
            status = "warning"
            check_failures = []
            check_warnings = issues or [f"CAD {level} preflight warning."]
            warnings.extend(check_warnings)
        elif status_value == "pass":
            status = "pass"
            check_failures = []
            check_warnings = []
        elif status_value == "not_evaluated":
            status = "not_evaluated"
            check_failures = []
            check_warnings = []
        else:
            status = "warning"
            warning = (
                f"CAD {level} preflight returned an unrecognized status: "
                f"{status_value!r}."
            )
            check_failures = []
            check_warnings = issues or [warning]
            warnings.extend(check_warnings)
        checks.append(
            ValidationCheck(
                name=f"cad_preflight.{level}",
                status=status,
                summary=str(
                    report.get("message") or f"CAD {level} preflight completed."
                ),
                failures=check_failures,
                warnings=check_warnings,
                evidence_artifacts=artifacts,
                metadata={
                    "profile_intent": profile_id,
                    "raw_status": report.get("strict_topology_report", {}).get("status", status_value),
                    "claim_scope": "source_preserving_render_only"
                    if preserved_render_reference is not None and level == "mesh_topology"
                    else "legacy_cad_preflight_only"
                    if level != "mesh_topology"
                    else "cad_preflight_only",
                    "required_by_profile": level in required_levels,
                    "final_domain_claim": False,
                },
            )
        )

    if delegated_levels:
        checks.append(
            ValidationCheck(
                name="cad_preflight.downstream_physics_delegation",
                status="not_evaluated",
                summary=(
                    "CAD physics/material preflight was not run; final checks are "
                    "delegated to content-workflow-physics."
                ),
                evidence_artifacts=artifacts,
                metadata={
                    "profile_intent": profile_id,
                    "delegated_levels": delegated_levels,
                    "delegate": "content-workflow-physics",
                    "compatibility_option": "run_legacy_cad_physics_preflight",
                },
            )
        )
    return checks, failures, warnings, artifacts, report_path


def _optimization_check(
    optimization: dict[str, Any],
) -> tuple[ValidationCheck, list[str], list[EvidenceArtifact]]:
    status_value = str(optimization.get("status") or "")
    degraded_reason = str(optimization.get("degraded_reason") or "")
    metadata_path = optimization.get("metadata_path")
    artifacts = (
        [
            EvidenceArtifact(
                kind="optimization_metadata",
                path=str(metadata_path),
                description="Shared OptimizeUSDTask metadata and Geometry policy.",
                metadata={
                    "producer": "world_understanding.agentic.usd_tasks.OptimizeUSDTask",
                    "claim_scope": "geometry_optimization",
                    "status": status_value,
                },
            )
        ]
        if metadata_path
        else []
    )
    metadata: dict[str, Any] = {
        "backend": str(optimization.get("backend") or ""),
        "requested_backend": str(optimization.get("requested_backend") or ""),
        "actual_backend": str(optimization.get("actual_backend") or ""),
        "fallback_used": bool(optimization.get("fallback_used")),
        "fallback_reason": optimization.get("fallback_reason"),
        "policy": str(optimization.get("policy") or ""),
        "status": status_value,
        "metadata_path": str(metadata_path or ""),
        "artifact_role": str(optimization.get("artifact_role") or ""),
        "missing_asset_dependencies": list(
            optimization.get("missing_asset_dependencies") or []
        ),
    }
    if degraded_reason:
        metadata["degraded_reason"] = degraded_reason
    if optimization.get("geometry_fidelity"):
        metadata["geometry_fidelity"] = optimization["geometry_fidelity"]
    if optimization.get("semantic_prim_boundaries"):
        metadata["semantic_prim_boundaries"] = optimization["semantic_prim_boundaries"]
    if optimization.get("rejected_output_usd"):
        metadata["rejected_output_usd"] = str(optimization["rejected_output_usd"])

    missing_asset_dependencies = metadata["missing_asset_dependencies"]
    dependency_warnings = (
        [
            "Geometry preserved unresolved appearance dependencies for downstream "
            "material/render repair: "
            + ", ".join(str(item) for item in missing_asset_dependencies)
        ]
        if missing_asset_dependencies
        else []
    )

    def result(
        check: ValidationCheck,
    ) -> tuple[ValidationCheck, list[str], list[EvidenceArtifact]]:
        if not dependency_warnings:
            return check, list(check.warnings), artifacts
        warnings = [*check.warnings, *dependency_warnings]
        status = "warning" if check.status == "pass" else check.status
        return (
            check.model_copy(update={"status": status, "warnings": warnings}),
            warnings,
            artifacts,
        )

    if status_value in {
        "optimization_unavailable",
        "fidelity_fallback",
        "semantic_fidelity_fallback",
        "semantic_lock_skip",
    }:
        warning = (
            "Shared Scene Optimizer output failed the geometry fidelity gate; "
            "the output is a normalized source copy and is not claimed as "
            f"optimized: {degraded_reason}"
            if status_value in {"fidelity_fallback", "semantic_fidelity_fallback"}
            else (
                "Scene optimization was skipped to preserve locked semantic "
                f"face provenance: {degraded_reason}"
                if status_value == "semantic_lock_skip"
                else "Shared Scene Optimizer was unavailable; the output is a normalized "
                f"source copy and is not claimed as optimized: {degraded_reason}"
            )
        )
        return result(
            ValidationCheck(
                name="usd_cli_scene_optimizer",
                status="warning",
                summary=(
                    "Scene optimization failed the fidelity gate; source fidelity was preserved."
                    if status_value
                    in {"fidelity_fallback", "semantic_fidelity_fallback"}
                    else "Scene optimization was skipped to preserve semantic locks."
                    if status_value == "semantic_lock_skip"
                    else "Scene optimization was unavailable; source fidelity was preserved."
                ),
                warnings=[warning],
                evidence_artifacts=artifacts,
                repair_hints=[
                    "Install or repair the usd-cli Scene Optimizer backend, "
                    "rerun geometry optimization, then regenerate validation evidence."
                ],
                metadata=metadata,
            )
        )

    if status_value == "completed" and optimization.get("fallback_used"):
        warning = (
            "Shared Scene Optimizer completed through backend fallback "
            f"({metadata['requested_backend']} -> {metadata['actual_backend']}): "
            f"{metadata.get('fallback_reason') or 'unspecified reason'}"
        )
        return result(
            ValidationCheck(
                name="usd_cli_scene_optimizer",
                status="warning",
                summary="Shared OptimizeUSDTask completed through a different backend.",
                warnings=[warning],
                evidence_artifacts=artifacts,
                metadata=metadata,
            )
        )

    if status_value == "completed" and degraded_reason:
        warning = (
            "Shared Scene Optimizer completed with a reported degradation: "
            f"{degraded_reason}"
        )
        return result(
            ValidationCheck(
                name="usd_cli_scene_optimizer",
                status="warning",
                summary="Shared OptimizeUSDTask completed with degradation.",
                warnings=[warning],
                evidence_artifacts=artifacts,
                metadata=metadata,
            )
        )

    if status_value == "skipped":
        return result(
            ValidationCheck(
                name="usd_cli_scene_optimizer",
                status="pass",
                summary="Scene optimization was explicitly skipped by policy.",
                evidence_artifacts=artifacts,
                metadata=metadata,
            )
        )

    if status_value == "completed":
        return result(
            ValidationCheck(
                name="usd_cli_scene_optimizer",
                status="pass",
                summary="Shared OptimizeUSDTask completed.",
                evidence_artifacts=artifacts,
                metadata=metadata,
            )
        )

    warning = (
        "Shared Scene Optimizer returned an unrecognized status and the output "
        f"is not claimed as optimized: {status_value!r}."
    )
    return result(
        ValidationCheck(
            name="usd_cli_scene_optimizer",
            status="warning",
            summary="Scene optimization status was not recognized.",
            warnings=[warning],
            evidence_artifacts=artifacts,
            metadata=metadata,
        )
    )


def _inspection_check(
    inspection: dict[str, Any], inspection_path: Path
) -> tuple[ValidationCheck, list[str], list[EvidenceArtifact]]:
    unavailable = inspection.get("status") == "inspection_unavailable"
    error = str(inspection.get("error") or "")
    warning = "Shared usd-cli mesh inspection was unavailable" + (
        f": {error}" if error else "."
    )
    artifact_status = "unavailable" if unavailable else "pass"
    artifacts = [
        EvidenceArtifact(
            kind="geometry_inspection_report",
            path=str(inspection_path),
            description="Shared usd-cli geometry inspection report.",
            metadata={
                "producer": "usd_cli.physics_ops.inspect_mesh_candidates",
                "claim_scope": "geometry_inspection",
                "status": artifact_status,
            },
        )
    ]
    return (
        ValidationCheck(
            name="usd_cli_geometry_inspection",
            status="warning" if unavailable else "pass",
            summary=(
                "Shared usd-cli mesh inspection was unavailable."
                if unavailable
                else "Shared usd-cli mesh inspection completed."
            ),
            warnings=[warning] if unavailable else [],
            evidence_artifacts=artifacts,
            metadata={
                "status": artifact_status,
                "candidate_count": inspection.get("candidate_count"),
                "error": error or None,
            },
        ),
        [warning] if unavailable else [],
        artifacts,
    )


def _mesh_normalization_check(
    normalization: dict[str, Any],
) -> tuple[ValidationCheck, list[str], list[EvidenceArtifact]]:
    report_path = str(normalization.get("report_path") or "")
    status = str(normalization.get("status") or "warning")
    skipped = list(normalization.get("skipped") or [])
    warnings = [
        "Mesh normal authoring skipped "
        f"{item.get('prim_path') or 'an unnamed mesh'}: {item.get('reason') or 'unknown reason'}"
        for item in skipped
    ]
    artifacts = (
        [
            EvidenceArtifact(
                kind="mesh_normalization_report",
                path=report_path,
                description="Geometry-owned imported-mesh normal authoring report.",
                metadata={
                    "producer": "content_agent_workflows.geometry.scene_ops",
                    "claim_scope": "geometry_conformance",
                    "status": status,
                },
            )
        ]
        if report_path
        else []
    )
    return (
        ValidationCheck(
            name="imported_mesh_normalization",
            status="pass" if status == "pass" else "warning",
            summary=(
                "Missing imported-mesh normals were authored without changing topology."
                if status == "pass"
                else "Imported-mesh normal authoring completed with skipped meshes."
            ),
            warnings=warnings,
            evidence_artifacts=artifacts,
            metadata={
                "modified_mesh_count": normalization.get("modified_mesh_count"),
                "unchanged_mesh_count": normalization.get("unchanged_mesh_count"),
                "points_or_topology_changed": normalization.get(
                    "points_or_topology_changed"
                ),
            },
        ),
        warnings,
        artifacts,
    )


def _runtime_check(
    *,
    usd_path: Path,
    output_dir: Path,
    params: GeometryWorkflowInput,
) -> tuple[ValidationCheck, list[str], list[str], list[EvidenceArtifact], Path | None]:
    result = run_runtime_validation(
        RuntimeValidationRequest(
            asset_path=usd_path,
            output_dir=output_dir / "runtime",
            mode=params.runtime_validation_mode,
            engine=params.runtime_engine,
            duration_s=params.runtime_duration_s,
            dt=params.runtime_dt,
            sample_fps=params.runtime_sample_fps,
        )
    )
    failures = list(result.failures)
    warnings = list(result.warnings)
    report_path = Path(result.report_path).resolve() if result.report_path else None
    artifacts = (
        [
            EvidenceArtifact(
                kind="runtime_validation_report",
                path=str(report_path),
                description="Shared runtime-validation workflow report.",
                metadata={
                    "producer": "content_agent_workflows.runtime_validation",
                    "claim_scope": result.claim_scope,
                    "status": result.status,
                },
            )
        ]
        if report_path
        else []
    )
    artifacts.extend(
        EvidenceArtifact(
            kind=str(item.get("kind") or "runtime_artifact"),
            path=str(item.get("path")),
            description=str(item.get("description") or ""),
            metadata={
                "producer": "usd_cli.physics_ops.validate_runtime",
                "claim_scope": result.claim_scope,
                "status": result.status,
            },
        )
        for item in result.evidence_artifacts
        if item.get("path")
    )
    return (
        ValidationCheck(
            name="runtime_loadability",
            status=result.status,
            summary=(
                "Runtime validation was explicitly skipped."
                if result.status == "not_evaluated"
                else "Shared runtime validation completed."
            ),
            failures=failures,
            warnings=warnings,
            evidence_artifacts=artifacts,
            metadata={
                "engine": params.runtime_engine,
                "mode": result.mode,
                "claim_scope": result.claim_scope,
                "temporary_proxy_used": result.temporary_proxy_used,
                "runtime_validation_usd": result.runtime_validation_usd,
                "legacy_run_runtime_validation": params.run_runtime_validation,
            },
        ),
        failures,
        warnings,
        artifacts,
        report_path,
    )


def _repair_visual_review_required(result: GeometryRepairResult | None) -> bool:
    if result is None:
        return False
    try:
        certificate = json.loads(
            Path(result.certificate_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return True
    return bool(certificate.get("render_geometry_changed"))


def _render_evidence(
    *,
    usd_path: Path,
    output_dir: Path,
    params: GeometryWorkflowInput,
    required: bool = False,
) -> tuple[
    ValidationCheck | None,
    list[str],
    list[str],
    list[EvidenceArtifact],
    list[dict[str, Any]],
    Path | None,
]:
    if not params.render_evidence and not required:
        return None, [], [], [], [], None

    candidate_sha256_before_render: str | None = None
    if required:
        try:
            candidate_sha256_before_render = file_sha256(usd_path)
        except OSError:
            pass
    result = render_geometry_evidence(
        usd_path=usd_path,
        output_dir=output_dir / "render_evidence",
        preset=params.render_preset,
        backend=params.render_backend,
        remote_base_url=params.render_remote_base_url,
        remote_api_key=(
            params.render_remote_api_key.get_secret_value()
            if params.render_remote_api_key is not None
            else None
        ),
        remote_allow_unauthenticated_identity=(
            params.render_remote_allow_unauthenticated_identity
        ),
        image_width=1024,
        image_height=1024,
        ovrtx_mode=params.render_ovrtx_mode,
        ovrtx_num_sensor_updates=params.render_ovrtx_num_sensor_updates,
    )
    candidate_binding: GeometryRenderCandidateBinding | None = None
    candidate_binding_path: Path | None = None
    if required:
        try:
            candidate_sha256_after_render = file_sha256(usd_path)
        except OSError:
            candidate_sha256_after_render = None
        candidate_binding = verify_geometry_render_candidate_evidence(
            result,
            candidate_usd_path=usd_path,
            candidate_sha256_before_render=candidate_sha256_before_render,
            candidate_sha256_after_render=candidate_sha256_after_render,
        )
        candidate_binding_path = _write_json(
            output_dir / "render_evidence" / "geometry_render_candidate_binding.json",
            _as_json(candidate_binding),
        )
    issues = [
        *result.shared_render_issues,
        *[
            issue
            for validation in result.image_validation
            for issue in validation.get("issues") or []
        ],
    ]
    messages = [
        str(issue.get("message") or issue.get("code") or "Unknown render issue")
        for issue in issues
    ]
    if result.status == "fail" or (required and result.status == "unavailable"):
        failures = messages or ["Shared render evidence failed without diagnostics."]
        warnings: list[str] = []
    elif result.status == "unavailable":
        failures = []
        warnings = messages or ["Shared render evidence was unavailable."]
    else:
        failure_severities = {"error", "fail", "failure", "fatal"}
        failures = [
            message
            for issue, message in zip(issues, messages, strict=True)
            if str(issue.get("severity") or "").lower() in failure_severities
        ]
        warnings = [
            message
            for issue, message in zip(issues, messages, strict=True)
            if str(issue.get("severity") or "").lower() not in failure_severities
        ]
    if candidate_binding is not None:
        failures.extend(candidate_binding.failures)
    result_source_sha256 = getattr(result, "source_usd_sha256", None)
    # The shared render path stamps explicit renderer identity, but evidence
    # models constructed by identity-naive callers carry the field defaults
    # (renderer None, unverified). A local "ovrtx" backend *is* the shared
    # OVRTX path by construction, so default-identity local evidence keeps
    # its verified-local meaning; remote evidence never gets inferred trust.
    explicit_renderer = getattr(result, "renderer", None)
    explicit_identity_verified = bool(
        getattr(result, "renderer_identity_verified", False)
    )
    identity_naive_local = (
        explicit_renderer is None
        and not explicit_identity_verified
        and result.backend == "ovrtx"
    )
    renderer = explicit_renderer or ("ovrtx" if identity_naive_local else None)
    renderer_identity_verified = explicit_identity_verified or identity_naive_local
    verified_ovrtx = renderer == "ovrtx" and renderer_identity_verified
    if not verified_ovrtx:
        failures.append(
            "Geometry final evidence requires verified OVRTX provenance; the "
            f"renderer identity was {renderer!r}."
        )
    accepted_ovrtx = (
        verified_ovrtx
        and result.status == "pass"
        and result.shared_render_status == "completed"
        and bool(result.image_paths)
        and not failures
    )
    render_view_kind = (
        "ovrtx_render_view"
        if verified_ovrtx
        else "unverified_remote_render_view"
        if result.backend == "remote"
        else "diagnostic_render_view"
    )
    status: Literal["pass", "fail", "warning"] = (
        "fail"
        if result.status == "fail" or failures
        else "warning"
        if warnings or result.status == "unavailable"
        else "pass"
    )
    report_path = Path(result.report_path).resolve() if result.report_path else None
    render_mode = (
        getattr(result, "ovrtx_render_mode", None) or params.render_ovrtx_mode or "pt"
    )
    sensor_updates = (
        getattr(result, "ovrtx_num_sensor_updates", None)
        or params.render_ovrtx_num_sensor_updates
        or 64
    )
    active_aov = getattr(result, "active_aov", None)
    report_accepted = accepted_ovrtx and (
        candidate_binding is None or candidate_binding.status == "pass"
    )
    artifacts = (
        [
            EvidenceArtifact(
                kind="shared_render_evidence_report",
                path=str(report_path),
                description=(
                    "Accepted OVRTX render and image-validation report."
                    if report_accepted
                    else "Diagnostic render and image-validation report."
                ),
                metadata={
                    "producer": "world_understanding.validation.usd_rendering",
                    "claim_scope": (
                        "geometry_visual_evidence"
                        if report_accepted
                        else "render_diagnostics"
                    ),
                    "status": (
                        candidate_binding.status
                        if candidate_binding is not None
                        else result.status
                    ),
                    "transport_backend": result.backend,
                    "renderer": renderer,
                    "renderer_identity_verified": renderer_identity_verified,
                    "ovrtx_render_mode": render_mode,
                    "ovrtx_num_sensor_updates": sensor_updates,
                    "active_aov": active_aov,
                    "candidate_usd_sha256": (
                        candidate_binding.candidate_sha256_before_render
                        if candidate_binding is not None
                        else result_source_sha256
                    ),
                    "render_metadata_sha256": (
                        candidate_binding.render_metadata_sha256
                        if candidate_binding is not None
                        else None
                    ),
                },
            )
        ]
        if report_path
        else []
    )
    if candidate_binding_path is not None and candidate_binding is not None:
        artifacts.append(
            EvidenceArtifact(
                kind="ovrtx_candidate_binding",
                path=str(candidate_binding_path),
                description=(
                    "Admission receipt binding OVRTX metadata and images to the exact "
                    "changed-geometry candidate."
                ),
                metadata={
                    "producer": "content_agent_workflows.geometry.rendering",
                    "claim_scope": (
                        "geometry_visual_evidence"
                        if report_accepted
                        else "diagnostic_only"
                    ),
                    "status": candidate_binding.status,
                    "candidate_usd_sha256": (
                        candidate_binding.candidate_sha256_before_render
                    ),
                    "render_metadata_sha256": (
                        candidate_binding.render_metadata_sha256
                    ),
                    "evidence_report_sha256": (
                        candidate_binding.evidence_report_sha256
                    ),
                },
            )
        )
    verified_bindings = {
        binding.path: binding
        for binding in (
            candidate_binding.verified_image_bindings if candidate_binding else []
        )
    }
    for image_path in result.image_paths:
        binding = verified_bindings.get(image_path)
        accepted = accepted_ovrtx and (
            candidate_binding is None
            or (candidate_binding.status == "pass" and binding is not None)
        )
        artifacts.append(
            EvidenceArtifact(
                kind=render_view_kind,
                path=image_path,
                description=(
                    "Individually validated shared OVRTX render view."
                    if accepted
                    else (
                        "OVRTX render view retained as diagnostic evidence because "
                        "the acceptance contract did not pass."
                        if verified_ovrtx
                        else "Diagnostic render view without verified OVRTX provenance."
                    )
                ),
                metadata={
                    "producer": "world_understanding.validation.usd_rendering",
                    "claim_scope": (
                        "geometry_visual_evidence"
                        if accepted and accepted_ovrtx
                        else "diagnostic_only"
                    ),
                    "status": result.status if accepted else "fail",
                    "transport_backend": result.backend,
                    "renderer": renderer,
                    "renderer_identity_verified": renderer_identity_verified,
                    "candidate_usd_sha256": (
                        candidate_binding.candidate_sha256_before_render
                        if candidate_binding is not None
                        else result_source_sha256
                    ),
                    "image_sha256": binding.sha256 if binding is not None else None,
                    "render_metadata_sha256": (
                        candidate_binding.render_metadata_sha256
                        if candidate_binding is not None
                        else None
                    ),
                    "ovrtx_render_mode": render_mode,
                    "ovrtx_num_sensor_updates": sensor_updates,
                    "active_aov": active_aov,
                },
            )
        )
    if result.presentation_image_path:
        artifacts.append(
            EvidenceArtifact(
                kind="render_presentation_grid",
                path=result.presentation_image_path,
                description=(
                    "Derived presentation grid; accepted individual OVRTX "
                    "views are the underlying evidence."
                    if accepted_ovrtx
                    else "Derived diagnostic presentation grid; no acceptance claim."
                ),
                metadata={
                    "producer": "content_agent_workflows.geometry.rendering",
                    "claim_scope": "presentation_only",
                    "status": result.status,
                    "source_renderer": result.backend,
                    "ovrtx_render_mode": render_mode,
                    "ovrtx_num_sensor_updates": sensor_updates,
                    "active_aov": active_aov,
                },
            )
        )
    preview_renders = (
        [
            {
                "kind": "render_presentation_grid",
                "path": result.presentation_image_path,
                "preset": params.render_preset,
                "backend": result.backend,
                "renderer": renderer,
                "renderer_identity_verified": renderer_identity_verified,
                "claim_scope": "presentation_only",
                "ovrtx_render_mode": render_mode,
                "ovrtx_num_sensor_updates": sensor_updates,
                "active_aov": active_aov,
            }
        ]
        if result.presentation_image_path
        else []
    )
    preview_renders.extend(
        {
            "kind": render_view_kind,
            "path": path,
            "preset": params.render_preset,
            "backend": result.backend,
            "renderer": renderer,
            "renderer_identity_verified": renderer_identity_verified,
            "claim_scope": (
                "accepted_render_evidence"
                if accepted_ovrtx
                and (
                    candidate_binding is None
                    or (
                        candidate_binding.status == "pass" and path in verified_bindings
                    )
                )
                else "diagnostic_only"
            ),
            "candidate_usd_sha256": (
                candidate_binding.candidate_sha256_before_render
                if candidate_binding is not None
                else result_source_sha256
            ),
            "image_sha256": (
                verified_bindings[path].sha256 if path in verified_bindings else None
            ),
            "render_metadata_sha256": (
                candidate_binding.render_metadata_sha256
                if candidate_binding is not None
                else None
            ),
            "ovrtx_render_mode": render_mode,
            "ovrtx_num_sensor_updates": sensor_updates,
            "active_aov": active_aov,
        }
        for path in result.image_paths
    )
    accepted_image_count = (
        sum(
            binding.role == "view"
            for binding in candidate_binding.verified_image_bindings
        )
        if report_accepted and candidate_binding is not None
        else len(result.image_paths)
        if report_accepted
        else 0
    )
    return (
        ValidationCheck(
            name="simulation_visual_review",
            status=status,
            summary=(
                "Required OVRTX repair evidence and generic image validation completed."
                if required
                else "Shared render evidence and generic image validation completed."
            ),
            failures=failures,
            warnings=warnings,
            evidence_artifacts=artifacts,
            metadata={
                "transport_backend": result.backend,
                "renderer": renderer,
                "renderer_identity_verified": renderer_identity_verified,
                "preset": params.render_preset,
                "shared_render_status": result.shared_render_status,
                "rendered_image_count": len(result.image_paths),
                "accepted_image_count": accepted_image_count,
                "diagnostic_image_count": len(result.image_paths)
                - accepted_image_count,
                "ovrtx_render_mode": render_mode,
                "ovrtx_num_sensor_updates": sensor_updates,
                "active_aov": active_aov,
                "candidate_binding": (
                    _as_json(candidate_binding)
                    if candidate_binding is not None
                    else None
                ),
            },
        ),
        failures,
        warnings,
        artifacts,
        preview_renders,
        report_path,
    )


def _accepted_render_image_path(
    preview_renders: list[dict[str, Any]],
) -> Path | None:
    """Select an accepted individual render view, never a presentation grid."""

    accepted = next(
        (
            preview
            for preview in preview_renders
            if preview.get("kind") == "ovrtx_render_view"
            and preview.get("claim_scope") == "accepted_render_evidence"
            and preview.get("path")
        ),
        None,
    )
    return Path(str(accepted["path"])).resolve() if accepted is not None else None


def _presentation_render_image_path(
    preview_renders: list[dict[str, Any]],
) -> Path | None:
    """Select the derived presentation grid for framing-only audit checks."""

    presentation = next(
        (
            preview
            for preview in preview_renders
            if preview.get("kind") == "render_presentation_grid"
            and preview.get("claim_scope") == "presentation_only"
            and preview.get("path")
        ),
        None,
    )
    return (
        Path(str(presentation["path"])).resolve() if presentation is not None else None
    )


def _shared_usd_check(
    *,
    usd_path: Path,
    output_dir: Path,
    enabled: bool,
) -> tuple[
    ValidationCheck,
    list[str],
    list[str],
    list[EvidenceArtifact],
    Path | None,
]:
    if not enabled:
        return (
            ValidationCheck(
                name="shared_usd_validation",
                status="not_evaluated",
                summary="Shared USD validation was explicitly disabled.",
            ),
            [],
            [],
            [],
            None,
        )
    report = run_geometry_usd_validation(usd_path, output_dir)
    report_path = Path(report.report_path).resolve() if report.report_path else None
    artifacts = (
        [
            EvidenceArtifact(
                kind="usd_validation_report",
                path=str(report_path),
                description="Shared NVIDIA USD validation report.",
                metadata={
                    "producer": report.producer,
                    "claim_scope": "usd_structure_and_geometry",
                    "status": report.status,
                },
            )
        ]
        if report_path
        else []
    )
    status: Literal["pass", "fail", "warning"] = (
        "pass"
        if report.status == "pass"
        else "warning"
        if report.status == "unavailable"
        else "fail"
    )
    return (
        ValidationCheck(
            name="shared_usd_validation",
            status=status,
            summary="Shared USD structural and geometry validation completed.",
            failures=list(report.failures),
            warnings=list(report.warnings),
            evidence_artifacts=artifacts,
            metadata={
                "producer": report.producer,
                "categories": report.categories,
                "upstream_status": report.status,
            },
        ),
        list(report.failures),
        list(report.warnings),
        artifacts,
        report_path,
    )


def _simready_check(
    *,
    usd_path: Path,
    output_dir: Path,
    params: GeometryWorkflowInput,
) -> tuple[
    ValidationCheck,
    list[str],
    list[str],
    list[EvidenceArtifact],
    Literal["pass", "conditional", "fail", "not_evaluated"],
    Path | None,
    Path | None,
]:
    if params.simready_mode == "skip":
        return (
            ValidationCheck(
                name="formal_simready_profile",
                status="not_evaluated",
                summary="Formal SimReady Foundation validation was not requested.",
            ),
            [],
            [],
            [],
            "not_evaluated",
            None,
            None,
        )

    simready_dir = output_dir / "simready"
    simready_dir.mkdir(parents=True, exist_ok=True)
    profile = resolve_official_simready_profile(
        local_profile_id=params.target_profile,
        official_profile=params.simready_profile,
    )
    report_path = simready_dir / "simready-profile.json"
    report = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=str(usd_path),
            profile=profile,
            profile_version=params.simready_profile_version,
            report_path=str(report_path),
            install_missing=params.simready_install_missing,
        )
    )
    validation_path = Path(report.report_path or report_path).resolve()
    artifacts = [
        EvidenceArtifact(
            kind="simready_foundation_validation",
            path=str(validation_path),
            description="Formal SimReady Foundation profile validation report.",
            metadata={
                "producer": "content_agent_workflows.simready",
                "claim_scope": "formal_simready_profile",
                "status": report.status,
                "profile": report.profile_target,
            },
        )
    ]
    failures: list[str] = []
    warnings: list[str] = []
    if report.status == "BLOCKED":
        status: Literal["pass", "fail", "warning"] = "warning"
        sim_ready_status: Literal["pass", "conditional", "fail", "not_evaluated"] = (
            "not_evaluated"
        )
        warnings.extend(
            report.errors or ["SimReady Foundation validation was blocked."]
        )
        warnings.extend(report.warnings)
    elif report.passed:
        status = "pass"
        sim_ready_status = "pass"
        warnings.extend(report.warnings)
    else:
        status = "fail"
        sim_ready_status = "fail"
        failures.extend(report.errors)
        failures.extend(
            str(item.get("message") or item.get("requirement") or "SimReady issue")
            for item in report.issues
        )
        if not failures:
            failures.append("Formal SimReady Foundation profile validation failed.")

    conformance_path: Path | None = None
    if (
        params.simready_mode == "validate_and_route_conformance"
        and report.status != "BLOCKED"
        and not report.passed
    ):
        conformance_path = simready_dir / "simready-conformance.json"
        conformance = run_simready_profile_conformance(
            SimReadyConformanceInput(
                asset_path=str(usd_path),
                output_dir=str(simready_dir / "conformance"),
                profile=profile,
                profile_version=params.simready_profile_version,
                report_path=str(conformance_path),
                validation_report_path=str(validation_path),
                source_asset=str(params.source_path) if params.source_path else None,
            )
        )
        artifacts.append(
            EvidenceArtifact(
                kind="simready_conformance_routing",
                path=str(conformance_path),
                description="Shared SimReady conformance routing report.",
                metadata={
                    "producer": "content_agent_workflows.simready",
                    "claim_scope": "simready_repair_routing",
                    "status": conformance.status,
                },
            )
        )
        if conformance.status != "PASS":
            warnings.extend(conformance.warnings)

    return (
        ValidationCheck(
            name="formal_simready_profile",
            status=status,
            summary=(
                "Formal SimReady Foundation validation completed."
                if report.status != "BLOCKED"
                else "Formal SimReady Foundation validation was blocked by its runtime."
            ),
            failures=failures,
            warnings=warnings,
            evidence_artifacts=artifacts,
            metadata={
                "profile": report.profile_target,
                "foundation_status": report.status,
                "sim_ready_status": sim_ready_status,
            },
        ),
        failures,
        warnings,
        artifacts,
        sim_ready_status,
        validation_path,
        conformance_path,
    )


def _source_prep_artifacts(
    prepared: PreparedGeometrySource,
) -> list[EvidenceArtifact]:
    artifacts: list[EvidenceArtifact] = []
    preserved = prepared.metadata.get("lossless_gltf")
    if isinstance(preserved, dict) and Path(str(preserved.get("receipt", ""))).is_file():
        artifacts.append(EvidenceArtifact(
            kind="source_fidelity_report",
            path=str(preserved["receipt"]),
            description="Static glTF source vertex/index identity and dependency receipts.",
            metadata={"producer": "content_agent_workflows.geometry.lossless_gltf", "claim_scope": "source_render_geometry", "status": "pass"},
        ))
    conversion = prepared.metadata.get("conversion")
    if not isinstance(conversion, dict):
        return artifacts
    for kind, key in (
        ("conversion_probe", "probe_path"),
        ("conversion_report", "report_path"),
        ("conversion_validation", "validation_report_path"),
        ("conversion_manifest", "manifest_path"),
    ):
        path = conversion.get(key)
        if path and Path(str(path)).is_file():
            artifacts.append(
                EvidenceArtifact(
                    kind=kind,
                    path=str(path),
                    description="Shared source-to-USD conversion evidence.",
                    metadata={
                        "producer": "content_agent_workflows.convert_to_usd",
                        "claim_scope": "source_conversion",
                        "status": str(conversion.get("status") or "not_evaluated"),
                    },
                )
            )
    return artifacts


def _representation_metadata(
    prepared: PreparedGeometrySource | None,
) -> list[dict[str, Any]]:
    if prepared is None:
        return []
    records: list[dict[str, Any]] = []
    source_bundle = prepared.metadata.get("source_bundle")
    if isinstance(source_bundle, dict):
        bundled = source_bundle.get("representations")
        if isinstance(bundled, list):
            for value in bundled:
                if not isinstance(value, dict):
                    continue
                artifact = value.get("artifact")
                records.append(
                    {
                        **value,
                        "path": value.get("materialized_path"),
                        "sha256": artifact.get("sha256")
                        if isinstance(artifact, dict)
                        else None,
                        "byte_size": artifact.get("size_bytes")
                        if isinstance(artifact, dict)
                        else None,
                        "source_bundle_id": source_bundle.get("bundle_id"),
                        "source_revision": source_bundle.get("source_revision"),
                        "producer": source_bundle.get("producer"),
                    }
                )
    value = prepared.metadata.get("representations")
    if not isinstance(value, list):
        return records
    known = {
        (str(item.get("representation_id") or ""), str(item.get("path") or ""))
        for item in records
    }
    records.extend(
        dict(item)
        for item in value
        if isinstance(item, dict)
        and (
            str(item.get("representation_id") or ""),
            str(item.get("path") or ""),
        )
        not in known
    )
    return records


def _representations_by_role(
    representations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_role: dict[str, list[dict[str, Any]]] = {
        role: [] for role in ("design", "render", "collision", "semantic")
    }
    for representation in representations:
        role = {
            "design_exchange": "design",
            "render_geometry": "render",
            "collision_candidate": "collision",
        }.get(
            str(representation.get("role") or ""),
            str(representation.get("role") or ""),
        )
        by_role.setdefault(role or "other", []).append(dict(representation))
    return by_role


def _primary_representation_metadata(
    prepared: PreparedGeometrySource,
    representations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    prepared_path = (
        str(Path(prepared.prepared_usd_path).resolve())
        if prepared.prepared_usd_path
        else None
    )
    primary_role = str(prepared.metadata.get("representation_role") or "")
    for representation in representations:
        path = representation.get("path")
        if (
            path
            and str(Path(str(path)).resolve()) == prepared_path
            and (not primary_role or representation.get("role") == primary_role)
        ):
            return dict(representation)
    source_bundle = prepared.metadata.get("source_bundle")
    selected = (
        source_bundle.get("selected_representation")
        if isinstance(source_bundle, dict)
        else None
    )
    if isinstance(selected, dict):
        selected_id = selected.get("representation_id")
        for representation in representations:
            if representation.get("representation_id") == selected_id:
                return dict(representation)
    return dict(representations[0]) if representations else None


def _source_record_artifacts(
    prepared: PreparedGeometrySource,
) -> list[EvidenceArtifact]:
    artifacts: list[EvidenceArtifact] = []
    seen: set[Path] = set()
    source_bundle = prepared.metadata.get("source_bundle")
    bundle_identity: dict[str, Any] = {}
    if isinstance(source_bundle, dict):
        bundle_identity = {
            "bundle_id": source_bundle.get("bundle_id"),
            "source_revision": source_bundle.get("source_revision"),
            "producer": source_bundle.get("producer"),
        }
        manifest_path = source_bundle.get("manifest_path")
        if manifest_path and Path(str(manifest_path)).is_file():
            manifest = Path(str(manifest_path)).resolve()
            seen.add(manifest)
            artifacts.append(
                EvidenceArtifact(
                    kind="geometry_source_bundle",
                    path=str(manifest),
                    description="Immutable provider-neutral geometry source bundle.",
                    metadata={
                        "producer": "geometry_authoring_provider",
                        "claim_scope": "geometry_source_identity",
                        "status": "admitted",
                        **bundle_identity,
                        "manifest_sha256": source_bundle.get("manifest_sha256"),
                    },
                )
            )
    for kind, raw_path, claim_scope in (
        (
            "source_of_record",
            prepared.source_path,
            "authoritative_source_representation",
        ),
        (
            "original_source_input",
            prepared.original_input_path,
            "original_source_input",
        ),
    ):
        if not raw_path:
            continue
        source_path = Path(raw_path).resolve()
        if source_path in seen or not source_path.is_file():
            continue
        seen.add(source_path)
        artifacts.append(
            EvidenceArtifact(
                kind=kind,
                path=str(source_path),
                description="Geometry source provenance artifact.",
                metadata={
                    "producer": "content_agent_workflows.geometry.source_prep",
                    "claim_scope": claim_scope,
                    "status": "prepared",
                    "source_fidelity_tier": prepared.fidelity_tier,
                    **bundle_identity,
                },
            )
        )
    for representation in _representation_metadata(prepared):
        path_value = representation.get("path")
        if not path_value:
            continue
        representation_path = Path(str(path_value)).resolve()
        if representation_path in seen or not representation_path.is_file():
            continue
        seen.add(representation_path)
        role = str(representation.get("role") or "other")
        artifacts.append(
            EvidenceArtifact(
                kind=f"source_{role}_representation",
                path=str(representation_path),
                description=f"Digest-bound provider {role} representation.",
                metadata={
                    "producer": "geometry_authoring_provider",
                    "claim_scope": f"source_representation.{role}",
                    "status": "recorded",
                    "representation_id": representation.get("representation_id"),
                    "representation_role": role,
                    "artifact_sha256": representation.get("sha256"),
                    "artifact_byte_size": representation.get("byte_size"),
                    **bundle_identity,
                },
            )
        )
    if isinstance(source_bundle, dict):
        provenance_inputs = source_bundle.get("provenance_input_artifacts")
        if isinstance(provenance_inputs, list):
            for index, record in enumerate(provenance_inputs):
                if not isinstance(record, dict):
                    continue
                path_value = record.get("materialized_path")
                if not path_value:
                    continue
                artifact_path = Path(str(path_value)).resolve()
                if artifact_path in seen or not artifact_path.is_file():
                    continue
                seen.add(artifact_path)
                artifacts.append(
                    EvidenceArtifact(
                        kind="source_provenance_input",
                        path=str(artifact_path),
                        description="Digest-bound provider provenance input.",
                        metadata={
                            "producer": "geometry_authoring_provider",
                            "claim_scope": "source_provenance.input_artifact",
                            "status": "recorded",
                            "input_index": index,
                            "artifact_sha256": record.get("sha256"),
                            "artifact_byte_size": record.get("size_bytes"),
                            **bundle_identity,
                        },
                    )
                )
    return artifacts


def _conversion_report_paths(
    prepared: PreparedGeometrySource | None,
) -> tuple[str | None, str | None]:
    if prepared is None:
        return None, None
    conversion = prepared.metadata.get("conversion")
    if not isinstance(conversion, dict):
        return None, None
    probe_path = conversion.get("probe_path")
    report_path = conversion.get("report_path")
    return (
        str(probe_path) if probe_path else None,
        str(report_path) if report_path else None,
    )


def _asset_audit_check(
    report: GeometryAssetAuditReport,
    artifact_paths: dict[str, str],
) -> tuple[ValidationCheck, list[str], list[str], list[EvidenceArtifact]]:
    failures = [signal.summary for signal in report.signals if signal.blocking]
    warnings = [signal.summary for signal in report.signals if not signal.blocking]
    status: Literal["pass", "fail", "warning"]
    if failures:
        status = "fail"
    elif warnings:
        status = "warning"
    else:
        status = "pass"
    artifacts = [
        EvidenceArtifact(
            kind=kind,
            path=path,
            description=f"Geometry asset audit artifact: {kind}.",
            metadata={
                "producer": "content_agent_workflows.geometry.audit",
                "claim_scope": "cad_source_and_product_audit",
                "status": status,
            },
        )
        for kind, path in artifact_paths.items()
    ]
    return (
        ValidationCheck(
            name="geometry_asset_audit",
            status=status,
            summary="Deterministic CAD source/render audit completed.",
            failures=failures,
            warnings=warnings,
            repair_hints=[
                signal.repair_hint
                for signal in report.signals
                if signal.blocking and signal.repair_hint
            ],
            evidence_artifacts=artifacts,
            metadata={
                "schema_version": report.schema_version,
                "passed": report.passed,
                "signal_codes": [signal.code for signal in report.signals],
            },
        ),
        failures,
        warnings,
        artifacts,
    )


def _geometry_repair_check(
    result: GeometryRepairResult,
) -> tuple[ValidationCheck, list[str], list[str], list[EvidenceArtifact]]:
    """Compose repair evidence without expanding its geometry-only claim scope."""

    status: Literal["pass", "fail", "warning"] = (
        "pass"
        if result.outcome == "certified"
        else "warning"
        if result.outcome == "conditional"
        else "fail"
    )
    certificate: dict[str, Any] = {}
    try:
        loaded = json.loads(Path(result.certificate_path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            certificate = loaded
    except (OSError, json.JSONDecodeError):
        certificate = {}
    failures = [str(item) for item in certificate.get("blockers") or []]
    warnings = [str(item) for item in certificate.get("remaining_warnings") or []]
    if status == "fail" and not failures:
        failures = ["Geometry repair rejected every bounded candidate."]
    if status == "warning" and not warnings:
        warnings = ["Geometry repair completed with a conditional certificate."]
    artifact_specs = [
        ("geometry_repair_diagnosis", result.diagnosis_path),
        ("geometry_repair_plan", result.repair_plan_path),
        ("geometry_repair_attempt_ledger", result.attempts_path),
        ("geometry_repair_certificate", result.certificate_path),
        ("geometry_repair_manifest", result.manifest_path),
    ]
    if result.collision_usd_path:
        artifact_specs.append(
            ("geometry_repair_collision_usd", result.collision_usd_path)
        )
    for kind, path in (
        (
            "geometry_repair_source_format_validation",
            result.source_format_validation_path,
        ),
        ("geometry_repair_usd_intake", result.usd_intake_path),
        (
            "geometry_repair_dependency_localization",
            result.dependency_localization_path,
        ),
        ("geometry_repair_scalable_audit", result.scalable_audit_path),
        ("geometry_repair_correspondence", result.correspondence_path),
        ("geometry_repair_source_collision_audit", result.source_collision_audit_path),
        (
            "geometry_repair_protected_feature_candidates",
            result.protected_feature_candidates_path,
        ),
        ("geometry_repair_manifold_seam_analysis", result.manifold_seam_analysis_path),
        ("geometry_repair_advanced_profile", result.advanced_profile_evidence_path),
    ):
        if path:
            artifact_specs.append((kind, path))
    artifacts = [
        EvidenceArtifact(
            kind=kind,
            path=path,
            description=f"Geometry repair artifact: {kind}.",
            metadata={
                "producer": "geometry_repair",
                "claim_scope": result.claim_scope,
                "status": result.outcome,
            },
        )
        for kind, path in artifact_specs
        if Path(path).is_file()
    ]
    return (
        ValidationCheck(
            name="geometry_repair",
            status=status,
            summary=(
                f"Profile-driven geometry repair outcome: {result.outcome}. "
                "The claim does not include final physics, materials, articulation, or SimReady status."
            ),
            evidence_artifacts=artifacts,
            failures=failures,
            warnings=warnings,
            repair_hints=[
                "Inspect the repair diagnosis and candidate ledger; supply missing intent or an approved hard-repair worker."
            ]
            if status == "fail"
            else [],
            metadata={
                "claim_scope": result.claim_scope,
                "outcome": result.outcome,
                "accepted_attempts": [
                    attempt.attempt_id
                    for attempt in result.attempts
                    if attempt.status == "accepted"
                ],
            },
        ),
        failures,
        warnings,
        artifacts,
    )


def _effective_optimization_policy(
    requested_policy: GeometryOptimizationPolicy,
    optimization: dict[str, Any],
) -> GeometryOptimizationPolicy:
    """Report the policy that actually produced the handoff geometry."""

    if optimization.get("artifact_role") != "optimized_geometry":
        return "skip"
    policy = optimization.get("policy")
    if policy in {"skip", "preserve_correspondence", "runtime_efficiency"}:
        return cast(GeometryOptimizationPolicy, policy)
    return requested_policy


def _build_handoff_manifest(
    *,
    route: GeometryRouteDecision,
    params: GeometryWorkflowInput,
    prepared_source: PreparedGeometrySource,
    source_usd: Path,
    geometry_input_usd: Path,
    optimized_usd: Path,
    inspection: dict[str, Any],
    optimization: dict[str, Any],
    validation_evidence_path: Path,
    evidence_bundle_path: Path,
    sim_ready_status: str,
    report_paths: dict[str, str | None],
    preview_renders: list[dict[str, Any]],
    asset_audit_report: GeometryAssetAuditReport | None,
    asset_audit_paths: dict[str, str],
    repair_result: GeometryRepairResult | None,
    native_repair_result: GeometryRepairResult | None,
    segmentation: GeometrySegmentationHandoff,
    handoff_ready: GeometryHandoffReady,
) -> dict[str, Any]:
    metadata = prepared_source.metadata
    source_bundle = (
        cast(dict[str, Any], metadata["source_bundle"])
        if isinstance(metadata.get("source_bundle"), dict)
        else None
    )
    representations = _representation_metadata(prepared_source)
    primary_representation = _primary_representation_metadata(
        prepared_source, representations
    )
    representations_by_role = _representations_by_role(representations)
    repaired_optimizer_input = geometry_input_usd.resolve() != source_usd.resolve()
    segmented_optimizer_input = bool(
        segmentation.segmented_usd_path
        and geometry_input_usd.resolve()
        == Path(segmentation.segmented_usd_path).resolve()
    )
    articulation_hints = {
        "delegate": "content-workflow-articulation",
        **cast(dict[str, Any], metadata.get("articulation_hints") or {}),
    }
    explicit_physics_hints = metadata.get("physics_hints")
    physics_hints = {
        "delegate": "content-workflow-physics",
        "source_fidelity_tier": prepared_source.fidelity_tier,
        **(
            cast(dict[str, Any], explicit_physics_hints)
            if isinstance(explicit_physics_hints, dict)
            else {}
        ),
    }
    original_input = (
        Path(prepared_source.original_input_path)
        if prepared_source.original_input_path
        else None
    )
    brep_source = (
        str(original_input)
        if original_input
        and original_input.suffix.lower() in {".brep", ".iges", ".igs", ".step", ".stp"}
        else metadata.get("brep_source_path")
    )
    stage_info = inspection.get("stage_info")
    if not isinstance(stage_info, dict):
        stage_info = {}
    return {
        "schema_version": GEOMETRY_HANDOFF_MANIFEST_SCHEMA_VERSION,
        "compatibility": {
            "legacy_mesh_geometry_field": "mesh_usd",
            "legacy_brep_mesh_alias_removed": True,
            "brep_source_field": "brep_source",
            "primary_representation_fields": ["source_usd", "prepared_usd"],
            "primary_representation_behavior": (
                "Convenience only; representation role claims are authoritative under "
                "representations.by_role."
            ),
            "brep_usd_behavior": (
                "null unless a future workflow emits a real BREP USD; use "
                "brep_source for STEP/IGES/BREP provenance"
            ),
        },
        "workflow": "geometry",
        "source_route": _as_json(route),
        "source_fidelity": _as_json(prepared_source),
        "source_bundle": source_bundle,
        "representations": {
            "primary": primary_representation,
            "artifacts": representations,
            "by_role": representations_by_role,
        },
        "source_usd": str(source_usd),
        "prepared_usd": str(source_usd),
        "geometry_usd": str(optimized_usd),
        "mesh_usd": str(optimized_usd),
        "workflow_geometry": {
            "role": "geometry_workflow_output",
            "path": str(optimized_usd),
            "sha256": file_sha256(optimized_usd),
            "byte_size": optimized_usd.stat().st_size,
            "format": optimized_usd.suffix.lower().lstrip("."),
            "derived_from": {
                "representation_id": primary_representation.get("representation_id")
                if primary_representation
                and not repaired_optimizer_input
                and not segmented_optimizer_input
                else None,
                "role": (
                    "semantic_segmentation"
                    if segmented_optimizer_input
                    else "geometry_repair_render"
                    if repaired_optimizer_input
                    else primary_representation.get("role")
                    if primary_representation
                    else None
                ),
                "path": str(geometry_input_usd.resolve()),
                "sha256": file_sha256(geometry_input_usd),
            },
            "role_claims": ["geometry"],
        },
        "brep_source": brep_source,
        "brep_usd": None,
        "optimization_status": str(optimization.get("status") or "unknown"),
        "optimization_policy": params.optimization_policy,
        "effective_optimization_policy": _effective_optimization_policy(
            params.optimization_policy,
            optimization,
        ),
        "units": {
            "up_axis": stage_info.get("up_axis"),
            "meters_per_unit": stage_info.get("meters_per_unit"),
        },
        "target_profile": params.target_profile,
        "target_runtime": params.target_runtime,
        "handoff_ready": handoff_ready,
        "segmentation": segmentation.manifest_record(),
        "semantic_parts": _semantic_parts_for_handoff(metadata, segmentation),
        "material_hints": list(metadata.get("material_hints") or []),
        "physics_hints": physics_hints,
        "articulation_hints": articulation_hints,
        "parameter_ranges": metadata.get("parameter_ranges") or {},
        "variant": {
            "variant_id": params.variant_id,
            "param_overrides": dict(params.param_overrides or {}),
            "source": (
                "source_bundle_parameter_overrides"
                if source_bundle
                else "runtime_parameter_overrides"
            )
            if params.param_overrides
            else "baseline",
        },
        "verification_assertions": metadata.get("verification_assertions") or [],
        "preview_renders": preview_renders,
        "asset_audit": {
            "passed": asset_audit_report.passed if asset_audit_report else None,
            "paths": asset_audit_paths,
            "blocking_signals": [
                signal.model_dump(mode="json")
                for signal in (asset_audit_report.signals if asset_audit_report else [])
                if signal.blocking
            ],
            "signals": [
                signal.model_dump(mode="json")
                for signal in (asset_audit_report.signals if asset_audit_report else [])
            ],
        },
        "geometry_repair": (
            {
                "outcome": repair_result.outcome,
                "claim_scope": repair_result.claim_scope,
                "diagnosis": repair_result.diagnosis_path,
                "source_format_validation": repair_result.source_format_validation_path,
                "repair_plan": repair_result.repair_plan_path,
                "attempt_ledger": repair_result.attempts_path,
                "certificate": repair_result.certificate_path,
                "manifest": repair_result.manifest_path,
                "render_usd": repair_result.render_usd_path,
                "collision_usd": repair_result.collision_usd_path,
                "usd_intake": repair_result.usd_intake_path,
                "dependency_localization": repair_result.dependency_localization_path,
                "scalable_audit": repair_result.scalable_audit_path,
                "correspondence": repair_result.correspondence_path,
                "source_collision_audit": repair_result.source_collision_audit_path,
                "protected_feature_candidates": (
                    repair_result.protected_feature_candidates_path
                ),
                "manifold_seam_analysis": repair_result.manifold_seam_analysis_path,
                "advanced_profile_evidence": repair_result.advanced_profile_evidence_path,
                "composed_usd": repair_result.composed_usd_path,
                "native_source_repair": (
                    {
                        "outcome": native_repair_result.outcome,
                        "claim_scope": native_repair_result.claim_scope,
                        "diagnosis": native_repair_result.diagnosis_path,
                        "repair_plan": native_repair_result.repair_plan_path,
                        "attempt_ledger": native_repair_result.attempts_path,
                        "certificate": native_repair_result.certificate_path,
                    }
                    if native_repair_result is not None
                    else None
                ),
            }
            if repair_result is not None
            else None
        ),
        "unresolved_geometry_issues": prepared_source.warnings + prepared_source.errors,
        "provenance": {
            "generated_by": "content_agent_workflows.geometry",
            "shared_optimization": optimization,
            "source_bundle_id": source_bundle.get("bundle_id")
            if source_bundle
            else None,
            "source_revision": source_bundle.get("source_revision")
            if source_bundle
            else None,
        },
        "build_stats": {
            "mesh_candidate_count": inspection.get("candidate_count"),
        },
        "validation_evidence": str(validation_evidence_path),
        "evidence_bundle": {
            "path": str(evidence_bundle_path),
            "sha256": file_sha256(evidence_bundle_path),
        },
        "reports": report_paths,
        "sim_ready_status": sim_ready_status,
        "runtime_validation_mode": params.runtime_validation_mode,
    }


def run_geometry_workflow(params: GeometryWorkflowInput) -> GeometryWorkflowResult:
    """Run the deterministic wrapper for geometry/CAD handoff preparation."""

    output_dir = Path(params.output_dir).resolve()
    route: GeometryRouteDecision | None = None
    segmentation_routing: GeometrySegmentationRoutingDecision | None = None
    segmentation_routing_path: Path | None = None
    prepared: PreparedGeometrySource | None = None
    native_repair_result: GeometryRepairResult | None = None
    try:
        verify_geometry_source_identity(
            source_path=params.source_path,
            source_manifest_path=params.source_manifest_path,
            expected_source_sha256=params.expected_source_sha256,
            expected_source_manifest_sha256=params.expected_source_manifest_sha256,
        )
        route = route_geometry_request(
            source_path=params.source_path,
            source_manifest_path=params.source_manifest_path,
            source_representation_role=params.source_representation_role,
            generated_usd_path=params.generated_usd_path,
            prompt=params.prompt,
            image_path=params.image_path,
        )
        if params.source_authoring_mode == "lossless_gltf":
            route = route.model_copy(update={
                "route": "provided_mesh_repair",
                "input_modality": "provided_mesh",
                "requires_scene_optimization": False,
                "rationale": "Explicit static glTF source-array preservation; no shared CAD conversion or render topology repair.",
            })
        output_dir.mkdir(parents=True, exist_ok=True)
        source_path_for_preparation = params.source_path
        source_sha256_for_preparation = params.expected_source_sha256
        derived_source_path_for_preparation: Path | None = None
        derived_source_sha256_for_preparation: str | None = None
        native_repair_source = params.source_path
        admitted_native_source = None
        native_representation_format: str | None = None
        if params.repair_mode != "off" and params.source_manifest_path is not None:
            admitted_native_source = admit_geometry_source_bundle(
                source_manifest_path=params.source_manifest_path,
                source_path=params.source_path,
                snapshot_parent=output_dir / "source_prep" / "admitted_bundles",
                source_representation_id=params.source_representation_id,
                source_representation_role=params.source_representation_role,
                generated_usd_path=params.generated_usd_path,
            )
            native_repair_source = admitted_native_source.selected_source
            native_representation_format = (
                admitted_native_source.selected_representation.format.lower()
            )
        if (
            params.repair_mode != "off"
            and native_repair_source is not None
            and Path(native_repair_source).suffix.lower() in BREP_SUFFIXES
            and (
                native_representation_format is None
                or native_representation_format in _NATIVE_BREP_FORMATS
            )
        ):
            native_repair_result = run_geometry_repair(
                RepairRequest(
                    source_path=native_repair_source,
                    output_dir=output_dir / "geometry_repair_native",
                    profile=params.repair_profile,
                    mode=params.repair_mode,
                    profile_confirmed=params.repair_profile_confirmed,
                    production_use=params.repair_production_use,
                    protected_features=params.repair_protected_features,
                    classified_holes=params.repair_classified_holes,
                    proposed_intents=params.repair_proposed_intents,
                    use_proposed_intent_ranking=params.repair_use_proposed_intent_ranking,
                    budgets=params.repair_budgets,
                    enabled_workers=params.repair_enabled_workers,
                    deterministic_seed=params.repair_deterministic_seed,
                    collision_runtime_engine=params.repair_collision_runtime_engine,
                    source_uri=params.repair_source_uri,
                    source_license=params.repair_source_license,
                    source_provenance=params.repair_source_provenance,
                    dependency_roots=params.repair_dependency_roots,
                    dependency_remap_manifest=params.repair_dependency_remap_manifest,
                    advanced_profile=params.repair_advanced_profile,
                )
            )
            if params.source_manifest_path is not None:
                assert admitted_native_source is not None
                readmitted_native_source = admit_geometry_source_bundle(
                    source_manifest_path=params.source_manifest_path,
                    source_path=params.source_path,
                    snapshot_parent=output_dir / "source_prep" / "admitted_bundles",
                    source_representation_id=params.source_representation_id,
                    source_representation_role=params.source_representation_role,
                    generated_usd_path=params.generated_usd_path,
                )
                if (
                    readmitted_native_source.manifest_sha256
                    != admitted_native_source.manifest_sha256
                    or readmitted_native_source.selected_source_sha256
                    != admitted_native_source.selected_source_sha256
                ):
                    raise RuntimeError(
                        "Geometry source bundle changed during native repair"
                    )
            else:
                verify_geometry_source_identity(
                    source_path=native_repair_source,
                    source_manifest_path=None,
                    expected_source_sha256=params.expected_source_sha256,
                    expected_source_manifest_sha256=None,
                )
            accepted_native_attempt = next(
                (
                    attempt
                    for attempt in native_repair_result.attempts
                    if attempt.status == "accepted"
                ),
                None,
            )
            if (
                accepted_native_attempt is not None
                and native_repair_result.outcome != "rejected"
            ):
                if accepted_native_attempt.output_sha256 is None:
                    raise RuntimeError(
                        "Accepted native repair output has no verified digest"
                    )
                if not accepted_native_attempt.output_path:
                    raise RuntimeError("Accepted native repair output is missing")
                accepted_native = Path(accepted_native_attempt.output_path).resolve()
                if not accepted_native.is_file():
                    raise RuntimeError("Accepted native repair output is missing")
                accepted_native_sha256 = file_sha256(accepted_native)
                if accepted_native_sha256 != accepted_native_attempt.output_sha256:
                    raise RuntimeError(
                        "Accepted native repair output changed after verification"
                    )
                if params.source_manifest_path is not None:
                    derived_source_path_for_preparation = accepted_native
                    derived_source_sha256_for_preparation = accepted_native_sha256
                else:
                    source_path_for_preparation = accepted_native
                    source_sha256_for_preparation = accepted_native_sha256
        prepared = prepare_geometry_source(
            source_path=source_path_for_preparation,
            source_manifest_path=params.source_manifest_path,
            source_representation_id=params.source_representation_id,
            source_representation_role=params.source_representation_role,
            generated_usd_path=params.generated_usd_path,
            prompt=params.prompt,
            image_path=params.image_path,
            output_dir=output_dir,
            source_authoring_mode=params.source_authoring_mode,
            allow_lossy_recovery=params.allow_lossy_recovery,
            param_overrides=params.param_overrides,
            install_missing_converters=params.install_missing_converters,
            converter_timeout_s=params.converter_timeout_s,
            expected_source_sha256=source_sha256_for_preparation,
            expected_source_manifest_sha256=params.expected_source_manifest_sha256,
            derived_source_path=derived_source_path_for_preparation,
            expected_derived_source_sha256=derived_source_sha256_for_preparation,
        )
        if not prepared.is_prepared or prepared.prepared_usd_path is None:
            detail = (
                "; ".join(prepared.errors) or "source preparation did not produce USD"
            )
            raise RuntimeError(f"Geometry source preparation failed: {detail}")
        source_usd = Path(prepared.prepared_usd_path).resolve()
        repair_result: GeometryRepairResult | None = None
        geometry_input_usd = source_usd
        if params.repair_mode != "off":
            repair_result = run_geometry_repair(
                RepairRequest(
                    source_path=source_usd,
                    output_dir=output_dir / "geometry_repair",
                    profile=params.repair_profile,
                    mode=params.repair_mode,
                    profile_confirmed=params.repair_profile_confirmed,
                    production_use=params.repair_production_use,
                    protected_features=params.repair_protected_features,
                    classified_holes=params.repair_classified_holes,
                    proposed_intents=params.repair_proposed_intents,
                    use_proposed_intent_ranking=params.repair_use_proposed_intent_ranking,
                    budgets=params.repair_budgets,
                    enabled_workers=params.repair_enabled_workers,
                    deterministic_seed=params.repair_deterministic_seed,
                    collision_runtime_engine=params.repair_collision_runtime_engine,
                    source_uri=params.repair_source_uri,
                    source_license=params.repair_source_license,
                    source_provenance=params.repair_source_provenance,
                    dependency_roots=params.repair_dependency_roots,
                    dependency_remap_manifest=params.repair_dependency_remap_manifest,
                    advanced_profile=params.repair_advanced_profile,
                )
            )
            if repair_result.render_usd_path:
                geometry_input_usd = Path(repair_result.render_usd_path).resolve()
        segmentation_required = params.segmentation_required or bool(
            params.segmentation_required_parts
        )
        supplied_semantic_parts = prepared.metadata.get("semantic_parts")
        segmentation_routing = route_segmentation(
            requested=params.segmentation_run_dir is not None or segmentation_required,
            source_usd_path=geometry_input_usd,
            completed_run_reference=params.segmentation_run_dir,
            required_semantic_names=params.segmentation_required_parts,
            supplied_semantic_parts=(
                supplied_semantic_parts
                if isinstance(supplied_semantic_parts, list)
                else None
            ),
        )
        segmentation_routing_path = _write_json(
            output_dir / "segmentation_routing.json",
            segmentation_routing.model_dump(mode="json"),
        )
        if segmentation_routing.route == "consume_completed_run":
            segmentation = consume_segmentation_handoff(
                run_dir=params.segmentation_run_dir,
                expected_source_usd=geometry_input_usd,
                required=segmentation_required,
                required_parts=params.segmentation_required_parts,
                require_ovrtx_evidence=params.segmentation_require_ovrtx_evidence,
            )
        else:
            segmentation = segmentation_handoff_from_routing(
                decision=segmentation_routing,
                expected_source_usd=geometry_input_usd,
                required=segmentation_required,
            )
        segmentation = segmentation.model_copy(
            update={
                "routing": segmentation_routing,
                "routing_decision_path": str(segmentation_routing_path),
            }
        )
        if segmentation.segmented_usd_path:
            geometry_input_usd = Path(segmentation.segmented_usd_path).resolve()
        effective_optimization_policy = params.optimization_policy
        if (
            segmentation_routing.route == "reuse_source_identity"
            and effective_optimization_policy != "skip"
        ):
            effective_optimization_policy = "skip"
            segmentation = segmentation.model_copy(
                update={
                    "warnings": [
                        *segmentation.warnings,
                        "Geometry skipped optimization to preserve source-authored "
                        "semantic prim and GeomSubset identity exactly.",
                    ]
                }
            )
        output_usd = (
            Path(params.output_usd_path).resolve()
            if params.output_usd_path
            else output_dir / f"{source_usd.stem}.geometry.usdc"
        )
        if geometry_input_usd == output_usd or source_usd == output_usd:
            output_usd = output_dir / f"{source_usd.stem}.geometry.usdc"
        optimization = scene_ops.optimize_geometry(
            source_usd=geometry_input_usd,
            output_usd=output_usd,
            policy=effective_optimization_policy,
            backend=params.optimizer_backend,
            optimization_config=params.optimization_config,
            protected_semantic_prim_paths=[
                part.output_prim_path for part in segmentation.parts
            ],
            expected_source_sha256=(
                segmentation.segmented_usd_sha256
                if segmentation.segmented_usd_path
                else file_sha256(geometry_input_usd)
            ),
        )
        coordinate_application = prepared.metadata.get("coordinate_system_application")
        if isinstance(coordinate_application, dict):
            declared_coordinates = coordinate_application.get("declared")
            if not isinstance(declared_coordinates, dict):
                raise RuntimeError(
                    "Prepared provider coordinates are missing their declaration"
                )
            prepared_usd_sha256 = prepared.metadata.get("prepared_usd_sha256")
            if not isinstance(prepared_usd_sha256, str):
                raise RuntimeError(
                    "Prepared provider coordinates are missing their source digest"
                )
            coordinate_retention = scene_ops.retain_geometry_source_coordinate_system(
                output_usd,
                output_dir / "geometry_source_coordinates.json",
                coordinate_system=declared_coordinates,
                immutable_source_usd=source_usd,
                expected_immutable_source_sha256=prepared_usd_sha256,
            )
            optimization["source_coordinate_system_retention"] = coordinate_retention
            metadata_path = optimization.get("metadata_path")
            if isinstance(metadata_path, str) and metadata_path:
                _write_json(
                    Path(metadata_path),
                    {
                        key: value
                        for key, value in optimization.items()
                        if key != "metadata_path"
                    },
                )
        if params.canonicalize_stage_metrics:
            metric_normalization = scene_ops.canonicalize_usd_stage_metrics(
                output_usd,
                output_dir / "geometry_stage_metrics.json",
            )
            optimization["stage_metric_normalization"] = metric_normalization
            metadata_path = optimization.get("metadata_path")
            if isinstance(metadata_path, str) and metadata_path:
                _write_json(
                    Path(metadata_path),
                    {
                        key: value
                        for key, value in optimization.items()
                        if key != "metadata_path"
                    },
                )
        effective_optimization_policy = _effective_optimization_policy(
            params.optimization_policy,
            optimization,
        )
        mesh_normalization = scene_ops.author_missing_mesh_normals(
            output_usd,
            output_dir / "mesh_normalization.json",
        )
        inspection = scene_ops.inspect_usd_geometry(output_usd)
        inspection_path = _write_json(
            output_dir / "geometry_inspection.json", inspection
        )

        checks, failures, warnings, artifacts, cad_preflight_report_path = (
            _cad_preflight_checks(
                output_usd,
                output_dir,
                params.target_profile,
                include_legacy_physics=params.run_legacy_cad_physics_preflight,
                preserved_render_reference=(
                    Path(prepared.prepared_usd_path)
                    if params.render_topology_policy == "preserve_source"
                    and prepared.fidelity_tier == "source_gltf_preserved"
                    else None
                ),
            )
        )
        source_prep_artifacts = _source_prep_artifacts(prepared)
        artifacts.extend(source_prep_artifacts)
        artifacts.extend(_source_record_artifacts(prepared))
        warnings.extend(prepared.warnings)
        failures.extend(prepared.errors)
        segmentation_check, segmentation_artifacts = segmentation_validation_check(
            segmentation
        )
        checks.append(segmentation_check)
        failures.extend(segmentation.failures)
        warnings.extend(segmentation.warnings)
        artifacts.extend(segmentation_artifacts)
        if repair_result is not None:
            repair_check, repair_failures, repair_warnings, repair_artifacts = (
                _geometry_repair_check(repair_result)
            )
            checks.append(repair_check)
            failures.extend(repair_failures)
            warnings.extend(repair_warnings)
            artifacts.extend(repair_artifacts)
        if native_repair_result is not None:
            native_specs = [
                (
                    "native_geometry_repair_diagnosis",
                    native_repair_result.diagnosis_path,
                ),
                ("native_geometry_repair_plan", native_repair_result.repair_plan_path),
                (
                    "native_geometry_repair_certificate",
                    native_repair_result.certificate_path,
                ),
            ]
            artifacts.extend(
                EvidenceArtifact(
                    kind=kind,
                    path=path,
                    description=f"Native CAD source repair artifact: {kind}.",
                    metadata={
                        "producer": "geometry_repair",
                        "claim_scope": native_repair_result.claim_scope,
                        "status": native_repair_result.outcome,
                    },
                )
                for kind, path in native_specs
                if Path(path).is_file()
            )
            if native_repair_result.outcome == "rejected":
                failures.append(
                    "Native CAD source repair rejected every bounded candidate."
                )
        artifacts.append(
            EvidenceArtifact(
                kind="geometry_usd",
                path=str(output_usd),
                description="Geometry workflow handoff USD.",
                metadata={
                    "producer": "content_agent_workflows.geometry",
                    "claim_scope": "geometry_handoff",
                    "status": str(optimization.get("status") or "unknown"),
                    "artifact_role": str(
                        optimization.get("artifact_role") or "geometry_output"
                    ),
                },
            )
        )
        (
            shared_usd_check,
            shared_usd_failures,
            shared_usd_warnings,
            shared_usd_artifacts,
            usd_validation_report_path,
        ) = _shared_usd_check(
            usd_path=output_usd,
            output_dir=output_dir,
            enabled=params.run_shared_usd_validation,
        )
        checks.append(shared_usd_check)
        failures.extend(shared_usd_failures)
        warnings.extend(shared_usd_warnings)
        artifacts.extend(shared_usd_artifacts)
        optimization_check, optimization_warnings, optimization_artifacts = (
            _optimization_check(optimization)
        )
        checks.append(optimization_check)
        warnings.extend(optimization_warnings)
        artifacts.extend(optimization_artifacts)
        (
            mesh_normalization_check,
            mesh_normalization_warnings,
            mesh_normalization_artifacts,
        ) = _mesh_normalization_check(mesh_normalization)
        checks.append(mesh_normalization_check)
        warnings.extend(mesh_normalization_warnings)
        artifacts.extend(mesh_normalization_artifacts)
        inspection_check, inspection_warnings, inspection_artifacts = _inspection_check(
            inspection, inspection_path
        )
        checks.append(inspection_check)
        warnings.extend(inspection_warnings)
        artifacts.extend(inspection_artifacts)
        (
            runtime_check,
            runtime_failures,
            runtime_warnings,
            runtime_artifacts,
            runtime_report_path,
        ) = _runtime_check(
            usd_path=output_usd,
            output_dir=output_dir,
            params=params,
        )
        checks.append(runtime_check)
        failures.extend(runtime_failures)
        warnings.extend(runtime_warnings)
        artifacts.extend(runtime_artifacts)
        (
            render_check,
            render_failures,
            render_warnings,
            render_artifacts,
            preview_renders,
            render_report_path,
        ) = _render_evidence(
            usd_path=output_usd,
            output_dir=output_dir,
            params=params,
            required=_repair_visual_review_required(repair_result),
        )
        if render_check is not None:
            checks.append(render_check)
        failures.extend(render_failures)
        warnings.extend(render_warnings)
        artifacts.extend(render_artifacts)

        asset_audit_report: GeometryAssetAuditReport | None = None
        asset_audit_paths: dict[str, str] = {}
        if params.run_asset_audit:
            audit_source_path = (
                Path(prepared.source_path) if prepared.source_path else None
            )
            render_image_path = _accepted_render_image_path(preview_renders)
            render_presentation_path = _presentation_render_image_path(preview_renders)
            audit_context = {
                **params.audit_context,
                "variant_id": params.variant_id,
                "param_overrides": dict(params.param_overrides or {}),
                "prepared_source_metadata": dict(prepared.metadata or {}),
                "source_fidelity_tier": prepared.fidelity_tier,
            }
            if (
                params.prompt
                and isinstance(prepared.metadata.get("source_bundle"), dict)
                and not audit_context.get("authoring_quality_checks")
            ):
                authoring_quality_checks = [
                    "parameter_ranges",
                    "semantic_parts",
                    "verify_probes",
                ]
                if _variant_audit_requested(params, audit_context):
                    authoring_quality_checks.append("variant_parameters")
                audit_context["authoring_quality_checks"] = authoring_quality_checks
            asset_audit_report = audit_geometry_asset(
                source_path=audit_source_path,
                prompt=params.prompt,
                context=audit_context,
                render_image_path=render_image_path,
                render_presentation_path=render_presentation_path,
                render_metadata_path=render_report_path,
                render_preset=params.render_preset,
                require_ovrtx_render=(
                    params.render_evidence
                    or _repair_visual_review_required(repair_result)
                ),
                trusted_artifact_roots=(output_dir,),
            )
            asset_audit_paths = write_audit_artifacts(
                output_dir=output_dir,
                report=asset_audit_report,
            )
            audit_check, audit_failures, audit_warnings, audit_artifacts = (
                _asset_audit_check(
                    asset_audit_report,
                    asset_audit_paths,
                )
            )
            checks.append(audit_check)
            failures.extend(audit_failures)
            warnings.extend(audit_warnings)
            artifacts.extend(audit_artifacts)

        (
            simready_check,
            simready_failures,
            simready_warnings,
            simready_artifacts,
            sim_ready_status,
            simready_report_path,
            simready_conformance_report_path,
        ) = _simready_check(
            usd_path=output_usd,
            output_dir=output_dir,
            params=params,
        )
        checks.append(simready_check)
        failures.extend(simready_failures)
        warnings.extend(simready_warnings)
        artifacts.extend(simready_artifacts)

        status = _validation_status(checks, failures, warnings)
        handoff_ready = _handoff_ready(status)
        source_bundle = (
            cast(dict[str, Any], prepared.metadata["source_bundle"])
            if isinstance(prepared.metadata.get("source_bundle"), dict)
            else None
        )
        representation_artifacts = _representations_by_role(
            _representation_metadata(prepared)
        )
        evidence = ValidationEvidence(
            workflow="geometry",
            asset=str(output_usd),
            target_runtime=params.target_runtime,
            validation_tier="T1_basic_stability",
            checks=checks,
            evidence_artifacts=artifacts,
            failures=failures,
            warnings=warnings,
            unresolved_issues=warnings if status == "conditional" else [],
            repair_hints=[
                "Repair CAD geometry, rerun usd-cli Scene Optimizer, then rerun SimReady/runtime validation."
            ]
            if status != "pass"
            else [],
            sim_ready_status=sim_ready_status,
            metadata={
                "schema_version": GEOMETRY_WORKFLOW_SCHEMA_VERSION,
                "route": route.route,
                "target_profile": params.target_profile,
                "formal_simready_mode": params.simready_mode,
                "runtime_validation_mode": params.runtime_validation_mode,
                "optimization_policy": params.optimization_policy,
                "effective_optimization_policy": effective_optimization_policy,
                "handoff_ready": handoff_ready,
                "segmentation": segmentation.manifest_record(),
                "repair_mode": params.repair_mode,
                "repair_profile": params.repair_profile,
                "repair_outcome": repair_result.outcome if repair_result else None,
                "repair_collision_runtime_engine": params.repair_collision_runtime_engine,
                "legacy_cad_physics_preflight": params.run_legacy_cad_physics_preflight,
                "legacy_run_runtime_validation": params.run_runtime_validation,
                "source_fidelity_tier": prepared.fidelity_tier,
                "source_bundle": source_bundle,
                "representation_artifacts": representation_artifacts,
                "variant_id": params.variant_id,
                "param_overrides": dict(params.param_overrides or {}),
            },
        )
        evidence_path = _write_json(
            output_dir / "geometry_validation_evidence.json", _as_json(evidence)
        )
        bundle_artifacts = [
            *artifacts,
            EvidenceArtifact(
                kind="geometry_validation_evidence",
                path=str(evidence_path),
                description="Geometry validation evidence composition.",
                metadata={
                    "producer": "content_agent_workflows.geometry",
                    "claim_scope": "geometry_handoff",
                    "status": status,
                },
            ),
        ]
        references = []
        seen_references: set[tuple[str, str]] = set()
        for artifact in bundle_artifacts:
            path = Path(artifact.path)
            key = (artifact.kind, str(path.resolve()))
            if key in seen_references or not path.is_file():
                continue
            seen_references.add(key)
            artifact_status = str(artifact.metadata.get("status") or "recorded")
            severity: Literal["info", "warning", "error"] = (
                "error"
                if artifact_status in {"fail", "failed", "error"}
                else "warning"
                if artifact_status
                in {
                    "warning",
                    "conditional",
                    "unavailable",
                    "optimization_unavailable",
                    "blocked",
                    "BLOCKED",
                }
                else "info"
            )
            references.append(
                evidence_reference(
                    kind=artifact.kind,
                    path=path,
                    producer=str(
                        artifact.metadata.get("producer")
                        or "content_agent_workflows.geometry"
                    ),
                    claim_scope=str(
                        artifact.metadata.get("claim_scope") or "geometry_handoff"
                    ),
                    status=artifact_status,
                    severity=severity,
                )
            )
        evidence_bundle = GeometryEvidenceBundle(
            source_asset=str(prepared.original_input_path or source_usd),
            geometry_usd=str(output_usd),
            geometry_validation_status=status,
            sim_ready_status=sim_ready_status,
            optimization_policy=params.optimization_policy,
            effective_optimization_policy=effective_optimization_policy,
            runtime_validation_mode=params.runtime_validation_mode,
            source_bundle=source_bundle,
            artifacts=references,
        )
        evidence_bundle_path = write_geometry_evidence_bundle(
            evidence_bundle, output_dir
        )
        conversion_probe_path, conversion_report_path = _conversion_report_paths(
            prepared
        )
        report_paths = {
            "conversion": conversion_report_path,
            "optimization": str(optimization.get("metadata_path") or "") or None,
            "mesh_normalization": str(mesh_normalization.get("report_path") or "")
            or None,
            "cad_preflight": str(cad_preflight_report_path),
            "usd_validation": str(usd_validation_report_path)
            if usd_validation_report_path
            else None,
            "runtime_validation": str(runtime_report_path)
            if runtime_report_path
            else None,
            "render_evidence": str(render_report_path) if render_report_path else None,
            "simready_validation": str(simready_report_path)
            if simready_report_path
            else None,
            "simready_conformance": str(simready_conformance_report_path)
            if simready_conformance_report_path
            else None,
            "asset_audit": asset_audit_paths.get("geometry_audit_path"),
            "geometry_repair": repair_result.certificate_path
            if repair_result
            else None,
            "part_segregation": segmentation.terminal_validation_path,
            "part_segregation_routing": str(segmentation_routing_path),
        }
        manifest = _build_handoff_manifest(
            route=route,
            params=params,
            prepared_source=prepared,
            source_usd=source_usd,
            geometry_input_usd=geometry_input_usd,
            optimized_usd=output_usd,
            inspection=inspection,
            optimization=optimization,
            validation_evidence_path=evidence_path,
            evidence_bundle_path=evidence_bundle_path,
            sim_ready_status=sim_ready_status,
            report_paths=report_paths,
            preview_renders=preview_renders,
            asset_audit_report=asset_audit_report,
            asset_audit_paths=asset_audit_paths,
            repair_result=repair_result,
            native_repair_result=native_repair_result,
            segmentation=segmentation,
            handoff_ready=handoff_ready,
        )
        manifest_path = _write_json(
            output_dir / "content_agents_manifest.json", manifest
        )
        if status == "fail" and params.fail_on_validation_error:
            raise RuntimeError(
                "Geometry validation failed; inspect "
                f"{evidence_path} and {manifest_path}."
            )
        return GeometryWorkflowResult(
            success=status != "fail",
            route=route,
            output_dir=str(output_dir),
            source_usd_path=str(source_usd),
            prepared_source_path=prepared.source_path,
            source_bundle_manifest_path=(
                str(source_bundle.get("manifest_path"))
                if source_bundle and source_bundle.get("manifest_path")
                else None
            ),
            source_bundle_id=(
                str(source_bundle.get("bundle_id")) if source_bundle else None
            ),
            source_revision=(
                str(source_bundle.get("source_revision")) if source_bundle else None
            ),
            source_provider_id=(
                str(cast(dict[str, Any], source_bundle["producer"]).get("provider_id"))
                if source_bundle and isinstance(source_bundle.get("producer"), dict)
                else None
            ),
            source_representation_id=(
                str(
                    cast(dict[str, Any], source_bundle["selected_representation"]).get(
                        "representation_id"
                    )
                )
                if source_bundle
                and isinstance(source_bundle.get("selected_representation"), dict)
                else None
            ),
            representation_artifacts=representation_artifacts,
            source_fidelity_tier=prepared.fidelity_tier,
            conversion_probe_path=conversion_probe_path,
            conversion_report_path=conversion_report_path,
            optimized_usd_path=str(output_usd),
            geometry_usd_path=str(output_usd),
            inspection_path=str(inspection_path),
            optimization_metadata_path=str(optimization.get("metadata_path") or ""),
            mesh_normalization_path=str(mesh_normalization.get("report_path") or "")
            or None,
            segmentation_outcome=segmentation.outcome,
            segmentation_route=segmentation_routing.route,
            segmentation_run_dir=segmentation.run_dir,
            segmentation_routing_path=str(segmentation_routing_path),
            segmentation_manifest_path=segmentation.producer_manifest_path,
            segmentation_usd_path=segmentation.segmented_usd_path,
            segmentation_validation_path=segmentation.terminal_validation_path,
            handoff_manifest_path=str(manifest_path),
            validation_evidence_path=str(evidence_path),
            runtime_report_path=str(runtime_report_path)
            if runtime_report_path
            else None,
            render_report_path=str(render_report_path) if render_report_path else None,
            usd_validation_report_path=str(usd_validation_report_path)
            if usd_validation_report_path
            else None,
            simready_report_path=str(simready_report_path)
            if simready_report_path
            else None,
            simready_conformance_report_path=str(simready_conformance_report_path)
            if simready_conformance_report_path
            else None,
            evidence_bundle_path=str(evidence_bundle_path),
            authoring_contract_path=asset_audit_paths.get("authoring_contract_path"),
            render_audit_path=asset_audit_paths.get("render_audit_path"),
            asset_audit_path=asset_audit_paths.get("geometry_audit_path"),
            asset_audit_summary_path=asset_audit_paths.get("asset_audit_summary_path"),
            repair_outcome=repair_result.outcome if repair_result else None,
            repair_diagnosis_path=repair_result.diagnosis_path
            if repair_result
            else None,
            repair_plan_path=repair_result.repair_plan_path if repair_result else None,
            repair_certificate_path=repair_result.certificate_path
            if repair_result
            else None,
            repair_manifest_path=repair_result.manifest_path if repair_result else None,
            repair_collision_usd_path=repair_result.collision_usd_path
            if repair_result
            else None,
            repair_usd_intake_path=repair_result.usd_intake_path
            if repair_result
            else None,
            repair_source_format_validation_path=(
                repair_result.source_format_validation_path if repair_result else None
            ),
            repair_dependency_localization_path=(
                repair_result.dependency_localization_path if repair_result else None
            ),
            repair_scalable_audit_path=repair_result.scalable_audit_path
            if repair_result
            else None,
            repair_correspondence_path=repair_result.correspondence_path
            if repair_result
            else None,
            repair_source_collision_audit_path=repair_result.source_collision_audit_path
            if repair_result
            else None,
            repair_protected_feature_candidates_path=(
                repair_result.protected_feature_candidates_path
                if repair_result
                else None
            ),
            repair_manifold_seam_analysis_path=(
                repair_result.manifold_seam_analysis_path if repair_result else None
            ),
            repair_advanced_profile_evidence_path=(
                repair_result.advanced_profile_evidence_path if repair_result else None
            ),
            native_repair_certificate_path=native_repair_result.certificate_path
            if native_repair_result
            else None,
            variant_id=params.variant_id,
            param_overrides=dict(params.param_overrides or {}),
            validation_status=status,
            handoff_ready=handoff_ready,
        )
    except Exception as exc:
        if params.fail_on_validation_error:
            raise
        traceback_path = None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            traceback_path = _write_json(output_dir / "geometry_failure.json", {
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "claim_scope": "workflow_failure_diagnostic",
            })
        except OSError:
            pass
        if route is None:
            route = GeometryRouteDecision(
                route="text_to_cad_generate",
                source_category="text_generation",
                input_modality="text",
                prompt=params.prompt,
                requires_generation=True,
                rationale="Geometry workflow failed before route selection completed.",
            )
        conversion_probe_path, conversion_report_path = _conversion_report_paths(
            prepared
        )
        return GeometryWorkflowResult(
            success=False,
            route=route,
            output_dir=str(output_dir),
            prepared_source_path=prepared.source_path if prepared else None,
            representation_artifacts=_representations_by_role(
                _representation_metadata(prepared)
            ),
            source_fidelity_tier=prepared.fidelity_tier if prepared else None,
            conversion_probe_path=conversion_probe_path,
            conversion_report_path=conversion_report_path,
            segmentation_route=(
                segmentation_routing.route if segmentation_routing is not None else None
            ),
            segmentation_routing_path=(
                str(segmentation_routing_path)
                if segmentation_routing_path is not None
                else None
            ),
            variant_id=params.variant_id,
            param_overrides=dict(params.param_overrides or {}),
            validation_status="fail",
            handoff_ready="no",
            error=str(exc),
            error_type=type(exc).__name__,
            error_traceback_path=str(traceback_path) if traceback_path else None,
        )
