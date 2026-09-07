# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Role-scoped geometry validation without weakening profile outcomes."""

from __future__ import annotations

from collections.abc import Iterable

from .models import (
    DiagnosisIssue,
    GeometryMetrics,
    GeometryRole,
    RepairProfile,
    RoleValidation,
    RoleValidationStatus,
)

_BREP_FORMATS = {"brep", "iges", "igs", "step", "stp"}
_ALL_ROLES: tuple[GeometryRole, ...] = ("render", "collision", "brep_source", "helper")


def required_roles_for_profile(
    profile: RepairProfile,
    *,
    source_format: str,
) -> set[GeometryRole]:
    """Return representation roles needed to complete the selected geometry profile."""

    required: set[GeometryRole] = {"render"}
    if profile != "visual_only":
        required.add("collision")
    if source_format.lower().lstrip(".") in _BREP_FORMATS:
        required.add("brep_source")
    return required


def _role_status(
    *,
    required: bool,
    applicable: bool,
    blockers: list[str],
    warnings: list[str],
    evaluated: bool,
) -> RoleValidationStatus:
    if blockers:
        return "fail"
    if warnings:
        return "conditional"
    if evaluated:
        return "pass"
    if required or applicable:
        return "not_evaluated"
    return "not_applicable"


def build_diagnosis_role_validation(
    *,
    profile: RepairProfile,
    metrics: GeometryMetrics,
    issues: Iterable[DiagnosisIssue],
    diagnosis_path: str | None,
    blocking_issue_ids: Iterable[str] | None = None,
    verified_issue_ids: Iterable[str] = (),
) -> dict[GeometryRole, RoleValidation]:
    """Project source diagnosis into role states while preserving aggregate behavior."""

    issue_list = list(issues)
    explicit_blockers = set(blocking_issue_ids) if blocking_issue_ids is not None else None
    verified = set(verified_issue_ids)
    required = required_roles_for_profile(profile, source_format=metrics.source_format)
    if any("brep_source" in issue.affected_roles for issue in issue_list):
        required.add("brep_source")
    result: dict[GeometryRole, RoleValidation] = {}
    for role in _ALL_ROLES:
        scoped = [issue for issue in issue_list if role in issue.affected_roles]
        blocking = [
            issue
            for issue in scoped
            if issue.issue_id not in verified
            and (
                issue.issue_id in explicit_blockers
                if explicit_blockers is not None
                else profile in issue.blocking_profiles
            )
        ]
        warning_issues = [
            issue
            for issue in scoped
            if issue.issue_id not in verified and issue.severity == "warning"
        ]
        applicable = bool(scoped)
        if role == "render":
            evaluated = metrics.mesh.mesh_count > 0
            applicable = applicable or evaluated
        elif role == "collision":
            # Source diagnosis deliberately does not accept generated or authored
            # collision. The dedicated source-collision and final collision audits
            # become authoritative later in the orchestrator.
            evaluated = False
            applicable = applicable or role in required or bool(metrics.source_collision_paths)
        elif role == "brep_source":
            evaluated = not blocking and (
                metrics.source_format.lower().lstrip(".") in _BREP_FORMATS
            )
            applicable = applicable or evaluated
        else:
            evaluated = False
            applicable = applicable or bool(metrics.source_helper_paths)
        result[role] = RoleValidation(
            role=role,
            required=role in required,
            status=_role_status(
                required=role in required,
                applicable=applicable,
                blockers=[issue.issue_id for issue in blocking],
                warnings=[issue.issue_id for issue in warning_issues],
                evaluated=evaluated,
            ),
            affected_prim_paths=sorted(
                {path for issue in scoped for path in issue.affected_prim_paths}
            ),
            issue_ids=sorted({issue.issue_id for issue in scoped}),
            evidence_paths=[diagnosis_path] if diagnosis_path else [],
            blockers=sorted({issue.issue_id for issue in blocking}),
            warnings=sorted({issue.issue_id for issue in warning_issues}),
        )
    return result


def _validation_from_status(
    *,
    role: GeometryRole,
    required: bool,
    status: RoleValidationStatus,
    base: RoleValidation | None,
    evidence_paths: Iterable[str],
    blockers: Iterable[str] = (),
    warnings: Iterable[str] = (),
    retain_base_blockers: bool = True,
    retain_base_warnings: bool = True,
) -> RoleValidation:
    combined_blockers = sorted(
        {
            *(base.blockers if base and retain_base_blockers else []),
            *blockers,
        }
    )
    combined_warnings = sorted(
        {
            *(base.warnings if base and retain_base_warnings else []),
            *warnings,
        }
    )
    if status == "pass":
        combined_blockers = []
    return RoleValidation(
        role=role,
        required=required,
        status=status,
        affected_prim_paths=list(base.affected_prim_paths if base else []),
        issue_ids=list(base.issue_ids if base else []),
        evidence_paths=sorted(
            {
                *(base.evidence_paths if base else []),
                *(path for path in evidence_paths if path),
            }
        ),
        blockers=combined_blockers,
        warnings=combined_warnings,
    )


def build_final_role_validation(
    *,
    profile: RepairProfile,
    diagnosis_role_validation: dict[GeometryRole, RoleValidation],
    source_format: str,
    render_path: str | None,
    collision_path: str | None,
    source_format_status: str,
    fidelity_status: str,
    visual_review_required: bool,
    collision_status: str,
    collision_runtime_status: str,
    usd_package_status: str,
    advanced_profile_status: str,
    evidence_paths_by_role: dict[GeometryRole, list[str]],
) -> dict[GeometryRole, RoleValidation]:
    """Derive final per-role states from the same evidence used by the certificate."""

    required = required_roles_for_profile(profile, source_format=source_format)
    brep_base = diagnosis_role_validation.get("brep_source")
    if brep_base and brep_base.status != "not_applicable":
        required.add("brep_source")
    result: dict[GeometryRole, RoleValidation] = {}

    render_base = diagnosis_role_validation.get("render")
    render_blockers = list(render_base.blockers if render_base else [])
    render_warnings = list(render_base.warnings if render_base else [])
    if source_format_status == "fail":
        render_blockers.append("source_format")
    elif source_format_status == "not_evaluated":
        render_warnings.append("source_format_not_evaluated")
    if fidelity_status == "fail":
        render_blockers.append("fidelity")
    elif fidelity_status in {"conditional", "not_evaluated"}:
        render_warnings.append(f"fidelity_{fidelity_status}")
    if usd_package_status == "fail":
        render_blockers.append("usd_package")
    elif usd_package_status == "conditional":
        render_warnings.append("usd_package_conditional")
    if visual_review_required:
        render_warnings.append("visual_review_required")
    if render_blockers:
        render_status: RoleValidationStatus = "fail"
    elif render_path is None:
        render_status = "not_evaluated"
    elif render_warnings:
        render_status = "conditional"
    else:
        render_status = "pass"
    result["render"] = _validation_from_status(
        role="render",
        required="render" in required,
        status=render_status,
        base=render_base,
        evidence_paths=evidence_paths_by_role.get("render", []),
        blockers=render_blockers,
        warnings=render_warnings,
    )

    collision_base = diagnosis_role_validation.get("collision")
    collision_has_validated_replacement = collision_path is not None and collision_status in {
        "pass",
        "conditional",
    }
    collision_blockers = (
        []
        if collision_has_validated_replacement
        else list(collision_base.blockers if collision_base else [])
    )
    collision_warnings = (
        []
        if collision_has_validated_replacement
        else list(collision_base.warnings if collision_base else [])
    )
    if collision_status == "fail":
        collision_blockers.append("collision_geometry")
    elif collision_status == "conditional":
        collision_warnings.append("collision_geometry_conditional")
    if collision_runtime_status == "fail":
        collision_blockers.append("collision_runtime")
    elif collision_runtime_status == "not_evaluated" and "collision" in required:
        collision_warnings.append("collision_runtime_not_evaluated")
    if collision_blockers:
        collision_role_status: RoleValidationStatus = "fail"
    elif collision_path is None:
        collision_role_status = "not_evaluated" if "collision" in required else "not_applicable"
    elif collision_warnings:
        collision_role_status = "conditional"
    else:
        collision_role_status = "pass"
    result["collision"] = _validation_from_status(
        role="collision",
        required="collision" in required,
        status=collision_role_status,
        base=collision_base,
        evidence_paths=evidence_paths_by_role.get("collision", []),
        blockers=collision_blockers,
        warnings=collision_warnings,
        retain_base_blockers=not collision_has_validated_replacement,
        retain_base_warnings=not collision_has_validated_replacement,
    )

    brep_applicable = "brep_source" in required or bool(
        brep_base and brep_base.status != "not_applicable"
    )
    brep_status: RoleValidationStatus = (
        brep_base.status if brep_base and brep_applicable else "not_applicable"
    )
    result["brep_source"] = _validation_from_status(
        role="brep_source",
        required="brep_source" in required,
        status=brep_status,
        base=brep_base,
        evidence_paths=evidence_paths_by_role.get("brep_source", []),
    )

    helper_base = diagnosis_role_validation.get("helper")
    helper_warnings = list(helper_base.warnings if helper_base else [])
    if advanced_profile_status == "fail":
        helper_status: RoleValidationStatus = "fail"
        helper_blockers = ["advanced_profile_geometry"]
    elif advanced_profile_status == "conditional":
        helper_status = "conditional"
        helper_blockers = []
        helper_warnings.append("advanced_profile_geometry_conditional")
    elif advanced_profile_status == "pass":
        helper_status = "pass"
        helper_blockers = []
    else:
        helper_status = (
            helper_base.status
            if helper_base and helper_base.status != "not_evaluated"
            else "not_applicable"
        )
        helper_blockers = list(helper_base.blockers if helper_base else [])
    result["helper"] = _validation_from_status(
        role="helper",
        required=False,
        status=helper_status,
        base=helper_base,
        evidence_paths=evidence_paths_by_role.get("helper", []),
        blockers=helper_blockers,
        warnings=helper_warnings,
    )
    return result


__all__ = [
    "build_diagnosis_role_validation",
    "build_final_role_validation",
    "required_roles_for_profile",
]
