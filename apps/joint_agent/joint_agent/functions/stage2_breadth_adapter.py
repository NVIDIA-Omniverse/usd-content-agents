# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict opt-in Stage 2 adapter for source-backed Joint 0.6 breadth."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from world_understanding.functions.physics.joint_rigger import (
    PLAN_SCHEMA_VERSION,
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointLimitV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerPlanV1,
    JointTopologyV1,
    validate_joint_topology_plan,
)

from joint_agent.functions import candidate_edge_authoring
from joint_agent.functions.articulation_candidates import (
    STAGE2_SCHEMA_VERSION,
    Stage2ArticulationCandidate,
)
from joint_agent.functions.bound_source_projection import (
    SourceProjectionMessages,
    bound_source_projection,
)
from joint_agent.functions.joint_0_6_breadth import (
    SourceBackedBreadthProof,
    requires_direct_breadth_construction,
    validate_source_backed_breadth_proof,
)

_UNRESOLVED_VALUES = frozenset({"", "unknown", "none", "null", "n/a", "na"})
_PRIVATE_BREADTH_REASON_FIELD = "breadth_unresolved_reason_codes"

type _PrivateBreadthReasonCode = Literal[
    "spherical_axis_not_applicable",
    "spherical_scalar_limit_unsupported",
    "spherical_source_evidence_required",
    "source_backed_proof_required",
    "source_joint_type_requires_source_adapter",
    "stage1_projection_rejected",
]


class _Stage2BreadthCandidate(Stage2ArticulationCandidate):
    """Closed parse-only candidate model used solely by the opt-in adapter."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_joint_type: Literal["continuous"] | None = None
    breadth_unresolved_reason_codes: list[_PrivateBreadthReasonCode] = Field(
        default_factory=list,
        exclude=True,
    )

    @model_validator(mode="after")
    def _validate_breadth_fields(self) -> Self:
        if (
            "source_joint_type" in self.model_fields_set
            and self.source_joint_type is None
        ):
            raise ValueError("source_joint_type cannot be explicitly null")

        allowed_source_fields = {
            "axis_hint",
            "fixed_parent_prim",
            "motion_axis_world",
            "motion_type",
            "source_joint_type",
        }
        unsupported_source_fields = sorted(
            set(self.field_sources) - allowed_source_fields
        )
        if unsupported_source_fields:
            raise ValueError(
                "field_sources carries unsupported breadth fields: "
                + ", ".join(unsupported_source_fields)
            )
        return self

    @model_validator(mode="after")
    def _validate_axis_invariants(self) -> Self:
        """Relax only the inherited cardinal-axis policy for gated breadth."""

        if self.motion_axis_world is None:
            return self
        normalized_hint = self.axis_hint.strip().lower()
        if (
            normalized_hint in candidate_edge_authoring._AXIS_VECTORS
            or normalized_hint in _UNRESOLVED_VALUES
        ):
            return self
        raise ValueError(
            "motion_axis_world requires a cardinal or unresolved axis_hint"
        )


class _Stage2BreadthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_count: int = Field(ge=0)
    ready_candidate_count: int = Field(ge=0)
    review_required_candidate_count: int = Field(ge=0)
    total_predictions: int | None = Field(default=None, ge=0)
    joint_type_counts: dict[str, int] = Field(default_factory=dict)
    unresolved_axis_count: int | None = Field(default=None, ge=0)
    unresolved_parent_count: int | None = Field(default=None, ge=0)
    review_status_counts: dict[str, int] = Field(default_factory=dict)
    limit_readiness_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    source_structure_diagnostics: list[str] = Field(default_factory=list)


class _Stage2BreadthDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["joint-agent-stage2-v0"]
    summary: _Stage2BreadthSummary
    candidates: list[_Stage2BreadthCandidate]


@dataclass(frozen=True, slots=True)
class SourceBackedAdmission:
    """One parsed candidate admitted by the shared source-proof validator."""

    candidate: Stage2ArticulationCandidate
    proof: SourceBackedBreadthProof


def _private_breadth_reason_code(
    value: object,
) -> _PrivateBreadthReasonCode | None:
    if value == "spherical_axis_not_applicable":
        return "spherical_axis_not_applicable"
    if value == "spherical_scalar_limit_unsupported":
        return "spherical_scalar_limit_unsupported"
    if value == "spherical_source_evidence_required":
        return "spherical_source_evidence_required"
    if value == "source_backed_proof_required":
        return "source_backed_proof_required"
    if value == "source_joint_type_requires_source_adapter":
        return "source_joint_type_requires_source_adapter"
    if value == "stage1_projection_rejected":
        return "stage1_projection_rejected"
    return None


def _split_candidate_reason_codes(
    raw_candidate: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    """Split private reasons without widening the inherited public field."""

    candidate = dict(raw_candidate)
    if _PRIVATE_BREADTH_REASON_FIELD in candidate:
        raise JointRiggerContractError(
            "stage2_breadth_projection_invalid",
            f"Stage 2 candidates[{index}].{_PRIVATE_BREADTH_REASON_FIELD} "
            "is internal-only",
        )
    raw_reason_codes = candidate.get("unresolved_reason_codes")
    if not isinstance(raw_reason_codes, list):
        return candidate

    public_reason_codes: list[Any] = []
    private_reason_codes: list[_PrivateBreadthReasonCode] = []
    for code in raw_reason_codes:
        private_code = _private_breadth_reason_code(code)
        if private_code is None:
            public_reason_codes.append(code)
        else:
            private_reason_codes.append(private_code)
    candidate["unresolved_reason_codes"] = public_reason_codes
    if private_reason_codes:
        candidate[_PRIVATE_BREADTH_REASON_FIELD] = private_reason_codes
    return candidate


def _proof_payload(candidate: Stage2ArticulationCandidate) -> dict[str, Any]:
    """Recombine public and private review state for proof admission."""

    payload: dict[str, Any] = candidate.model_dump(mode="python")
    if isinstance(candidate, _Stage2BreadthCandidate):
        payload["unresolved_reason_codes"] = [
            *candidate.unresolved_reason_codes,
            *candidate.breadth_unresolved_reason_codes,
        ]
    return payload


def legacy_preflight_deferred_candidate_ids(
    candidates: Iterable[Stage2ArticulationCandidate],
) -> frozenset[str]:
    """Select rows whose exact bytes cannot enter the released 0.5 preflight.

    Direct Joint 0.6 topologies are constructed privately. Review-only rows
    carrying private reason codes also need a public-safe placeholder even
    when their cardinal topology would otherwise use the legacy builder.
    """

    return frozenset(
        candidate.candidate_id
        for candidate in candidates
        if requires_direct_breadth_construction(_proof_payload(candidate))
        or (
            isinstance(candidate, _Stage2BreadthCandidate)
            and bool(candidate.breadth_unresolved_reason_codes)
        )
    )


def direct_breadth_admissions(
    admissions: Iterable[SourceBackedAdmission],
) -> tuple[SourceBackedAdmission, ...]:
    """Keep proved admissions that require private direct construction."""

    return tuple(
        admission
        for admission in admissions
        if requires_direct_breadth_construction(_proof_payload(admission.candidate))
    )


def load_breadth_candidates(
    candidate_bytes: bytes,
    *,
    path: Path,
) -> tuple[Stage2ArticulationCandidate, ...]:
    """Parse one closed Joint 0.6 document without widening Stage2-v0."""

    return tuple(_load_breadth_document(candidate_bytes, path=path).candidates)


def _load_breadth_document(
    candidate_bytes: bytes,
    *,
    path: Path,
) -> _Stage2BreadthDocument:
    try:
        raw = json.loads(
            candidate_bytes,
            object_pairs_hook=candidate_edge_authoring._object_without_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise JointRiggerContractError(
            "stage2_artifact_invalid",
            f"cannot decode Stage 2 candidate document {path}: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raise JointRiggerContractError(
            "stage2_artifact_invalid",
            "Stage 2 candidate document must be a JSON object",
        )
    raw_candidates = raw.get("candidates")
    validation_raw = dict(raw)
    if isinstance(raw_candidates, list):
        validation_candidates: list[Any] = []
        for index, raw_candidate in enumerate(raw_candidates):
            if not isinstance(raw_candidate, dict):
                raise JointRiggerContractError(
                    "stage2_artifact_invalid",
                    f"Stage 2 candidates[{index}] must be a JSON object",
                )
            if raw_candidate.get("schema_version") != STAGE2_SCHEMA_VERSION:
                raise JointRiggerContractError(
                    "stage2_artifact_invalid",
                    f"Stage 2 candidates[{index}].schema_version must be "
                    f"{STAGE2_SCHEMA_VERSION!r}",
                )
            validation_candidates.append(
                _split_candidate_reason_codes(raw_candidate, index=index)
            )
        validation_raw["candidates"] = validation_candidates
    try:
        document = _Stage2BreadthDocument.model_validate(
            validation_raw,
            strict=True,
        )
    except ValidationError as exc:
        raise JointRiggerContractError(
            "stage2_breadth_projection_invalid",
            "Joint 0.6 breadth accepts only declared Stage 2 fields and exact "
            f"source_joint_type=continuous: {exc}",
        ) from exc

    candidate_ids = [candidate.candidate_id for candidate in document.candidates]
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise JointRiggerContractError(
            "stage2_artifact_invalid",
            "Stage 2 candidate_id values must be unique; duplicates: "
            + ", ".join(duplicate_ids),
        )

    ready_count = sum(
        candidate.review_status == "ready_for_rigger_input"
        for candidate in document.candidates
    )
    expected_summary = {
        "candidate_count": len(document.candidates),
        "ready_candidate_count": ready_count,
        "review_required_candidate_count": len(document.candidates) - ready_count,
        "joint_type_counts": dict(
            sorted(
                Counter(
                    candidate.joint_type_hint for candidate in document.candidates
                ).items()
            )
        ),
        "unresolved_axis_count": sum(
            candidate.motion_axis_world is None and candidate.motion_type != "spherical"
            for candidate in document.candidates
        ),
        "unresolved_parent_count": sum(
            candidate.fixed_parent_prim is None for candidate in document.candidates
        ),
        "review_status_counts": dict(
            sorted(
                Counter(
                    candidate.review_status for candidate in document.candidates
                ).items()
            )
        ),
        "limit_readiness_counts": dict(
            sorted(
                Counter(
                    candidate.limit_readiness for candidate in document.candidates
                ).items()
            )
        ),
        "reason_code_counts": dict(
            sorted(
                Counter(
                    code
                    for candidate in document.candidates
                    for code in (
                        *candidate.unresolved_reason_codes,
                        *candidate.breadth_unresolved_reason_codes,
                    )
                ).items()
            )
        ),
    }
    for field, expected in expected_summary.items():
        if field not in document.summary.model_fields_set:
            continue
        if getattr(document.summary, field) != expected:
            raise JointRiggerContractError(
                "stage2_artifact_invalid",
                f"Stage 2 summary.{field} does not match candidates ({expected})",
            )
    return document


def admit_ready_candidates(
    candidates: Iterable[Stage2ArticulationCandidate],
) -> tuple[SourceBackedAdmission, ...]:
    """Admit exactly the ready candidates accepted by the shared proof."""

    admitted: list[SourceBackedAdmission] = []
    for candidate in candidates:
        if candidate.review_status != "ready_for_rigger_input":
            continue
        result = validate_source_backed_breadth_proof(
            _proof_payload(candidate),
        )
        if not result.requested:
            continue
        if result.failures:
            failure = result.failures[0]
            raise JointRiggerContractError(failure.code, failure.detail)
        if result.proof is None:  # pragma: no cover - result invariant
            raise JointRiggerContractError(
                "stage2_source_candidate_not_ready",
                f"candidate {candidate.candidate_id!r} produced no source proof",
            )
        admitted.append(SourceBackedAdmission(candidate=candidate, proof=result.proof))
    return tuple(admitted)


def build_source_backed_joint_plans(
    *,
    source_path: Path,
    source_asset: ArtifactIdentityV1,
    candidate_artifact: ArtifactIdentityV1,
    admissions: tuple[SourceBackedAdmission, ...],
    legacy_joint_plans: tuple[JointPlanV1, ...] = (),
) -> tuple[JointPlanV1, ...]:
    """Adapt proved topologies and validate the exact combined owned plan."""

    if not admissions:
        return ()
    try:
        from pxr import Sdf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - legacy preflight also needs pxr
        raise JointRiggerContractError(
            "openusd_unavailable",
            "OpenUSD bindings are required for source-backed Joint 0.6 preflight",
        ) from exc

    with _bound_source_projection(
        source_path=source_path,
        source_asset=source_asset,
    ) as bound_source:
        try:
            stage = Usd.Stage.Open(str(bound_source))
        except Exception as exc:
            raise JointRiggerContractError(
                "stage2_source_invalid",
                f"cannot open bound source stage for {source_path}: {exc}",
            ) from exc
        if stage is None:
            raise JointRiggerContractError(
                "stage2_source_invalid",
                f"cannot open bound source stage for {source_path}",
            )
        try:
            plans = tuple(
                _source_backed_joint_plan(
                    stage=stage,
                    admission=admission,
                    candidate_artifact=candidate_artifact,
                    Sdf=Sdf,
                    Usd=Usd,
                    UsdGeom=UsdGeom,
                )
                for admission in admissions
            )
            validate_joint_topology_plan(
                stage,
                JointRiggerPlanV1(
                    schema_version=PLAN_SCHEMA_VERSION,
                    joints=(*legacy_joint_plans, *plans),
                ),
            )
        finally:
            del stage
    return plans


@contextmanager
def _bound_source_projection(
    *,
    source_path: Path,
    source_asset: ArtifactIdentityV1,
) -> Iterator[Path]:
    """Yield one descriptor-derived source closure and clean it deterministically."""

    with bound_source_projection(
        source_path=source_path,
        expected_source=source_asset,
        error_prefix="stage2",
        messages=SourceProjectionMessages(
            binding_mismatch=(
                "source USD or its dependency closure does not match source_asset"
            ),
            materialization_failure=(
                "cannot materialize the bound source dependency closure"
            ),
            changed_before_use=("bound source changed before Joint 0.6 preflight"),
            changed_during_use=("bound source changed during Joint 0.6 preflight"),
        ),
    ) as bound_source:
        yield bound_source


def _source_backed_joint_plan(
    *,
    stage: Any,
    admission: SourceBackedAdmission,
    candidate_artifact: ArtifactIdentityV1,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
) -> JointPlanV1:
    proof = admission.proof
    prefix = f"Stage 2 candidate {proof.candidate_id!r}"
    _require_source_endpoint(
        stage,
        proof.body0,
        label=f"{prefix} body0",
        Sdf=Sdf,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    for index, member_path in enumerate(proof.moving_part_prims):
        _require_source_endpoint(
            stage,
            member_path,
            label=f"{prefix} moving_part_prims[{index}]",
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
        )

    topology_provenance = {
        "joint_type": _proof_field_evidence(
            candidate_artifact,
            proof,
            field="motion_type",
            properties=(
                ("motion_type", "source_joint_type", "field_sources")
                if proof.source_joint_type == "continuous"
                else ("motion_type", "field_sources")
            ),
            prim_path=proof.body1,
        ),
        "body0": _proof_field_evidence(
            candidate_artifact,
            proof,
            field="fixed_parent_prim",
            properties=(
                "fixed_parent_prim",
                "parent_resolution_source",
                "connectivity_evidence",
                "field_sources",
            ),
            prim_path=proof.body0,
        ),
        "body1": _proof_field_evidence(
            candidate_artifact,
            proof,
            field="moving_part_prims",
            properties=(
                "moving_part_prims",
                "connectivity_evidence",
                "field_sources",
            ),
            prim_path=proof.body1,
        ),
    }
    if proof.axis is not None:
        topology_provenance["axis_stage"] = _proof_field_evidence(
            candidate_artifact,
            proof,
            field="axis_stage",
            properties=(
                "axis_hint",
                "motion_axis_world",
                "axis_evidence",
                "field_sources",
            ),
            prim_path=proof.body1,
        )

    limit: JointLimitV1 | None = None
    if proof.limit is not None:
        try:
            limit = JointLimitV1(
                lower=proof.limit.lower,
                upper=proof.limit.upper,
                unit=proof.limit.unit,
                provenance=_proof_field_evidence(
                    candidate_artifact,
                    proof,
                    field="limit",
                    properties=(
                        "limit_unit",
                        "lower_limit",
                        "upper_limit",
                        "limit_source",
                        "limit_readiness",
                        "limit_evidence",
                    ),
                    prim_path=proof.body1,
                ),
            )
        except ValueError as exc:  # pragma: no cover - shared proof invariant
            raise JointRiggerContractError(
                "stage2_limit_invalid",
                f"{prefix} has an invalid limit: {exc}",
            ) from exc

    return JointPlanV1(
        topology=JointTopologyV1(
            joint_id=proof.candidate_id,
            joint_type=proof.motion_type,
            body0=proof.body0,
            body1=proof.body1,
            axis_stage=proof.axis,
            field_provenance=topology_provenance,
        ),
        limit=limit,
    )


def _require_source_endpoint(
    stage: Any,
    path: str,
    *,
    label: str,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
) -> None:
    candidate_edge_authoring._validate_absolute_prim_path(path, label=label, sdf=Sdf)
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid() or not prim.IsActive() or not prim.IsDefined():
        raise JointRiggerContractError(
            "stage2_source_endpoint_missing",
            f"{label} does not resolve to an active, defined source prim: {path}",
        )
    if prim.IsInstanceProxy():
        raise JointRiggerContractError(
            "stage2_source_endpoint_instance_unsupported",
            f"{label} cannot target an instance proxy: {path}",
        )
    if not UsdGeom.Xformable(prim):
        raise JointRiggerContractError(
            "stage2_source_endpoint_not_transformable",
            f"{label} is not transformable: {path}",
        )
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        prim
    )
    try:
        candidate_edge_authoring._validate_invertible_transform(
            transform,
            label=label,
        )
    except ValueError as exc:
        raise JointRiggerContractError(
            "stage2_source_endpoint_transform_invalid",
            str(exc),
        ) from exc


def _proof_field_evidence(
    artifact: ArtifactIdentityV1,
    proof: SourceBackedBreadthProof,
    *,
    field: str,
    properties: tuple[str, ...],
    prim_path: str,
) -> FieldProvenanceV1:
    authorities = [
        f"motion={proof.motion_source}",
        f"parent={proof.parent_source}",
        f"connectivity={proof.connectivity_source}",
    ]
    if proof.axis_source is not None:
        authorities.append(f"axis={proof.axis_source}")
    if proof.limit is not None:
        authorities.append(f"limit={proof.limit.source}")
    return FieldProvenanceV1(
        source="accepted_manifest",
        artifact=artifact,
        prim_path=prim_path,
        properties=properties,
        derivation=proof.derivation,
        evidence=(
            f"Shared source proof admitted candidate {proof.candidate_id!r} "
            f"field {field!r}; " + ", ".join(authorities) + "."
        ),
    )


__all__ = [
    "SourceBackedAdmission",
    "admit_ready_candidates",
    "build_source_backed_joint_plans",
    "direct_breadth_admissions",
    "legacy_preflight_deferred_candidate_ids",
    "load_breadth_candidates",
]
