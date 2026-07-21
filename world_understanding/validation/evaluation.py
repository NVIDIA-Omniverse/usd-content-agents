# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared evidence-backed evaluation signal contracts.

These models are intentionally lower-level than Validation Agent verdicts.
They represent findings and candidate corrections. A harness, validation gate,
or application adapter decides whether those signals should stop, warn, retry,
or refine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from world_understanding.validation.json_normalization import (
    StructuredJsonNormalizer,
)
from world_understanding.validation.models import (
    ISSUE_CODE_PATTERN,
    IssueSeverity,
    ValidationEvidence,
    ValidationModel,
    field_validator,
)

EvaluationStatus = Literal["completed", "partial", "skipped", "error"]
EVALUATION_SCHEMA_VERSION: Literal["evaluation-signals/v1"] = "evaluation-signals/v1"
_JSON_NORMALIZER = StructuredJsonNormalizer(model_types=(ValidationModel,))
_json_mapping = _JSON_NORMALIZER.mapping
_json_value = _JSON_NORMALIZER.value


class EvaluationSubject(ValidationModel):
    """Entity that an evaluation finding or correction is about."""

    kind: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class EvaluationCorrection(ValidationModel):
    """Candidate field-level correction emitted by an evaluator."""

    subject: EvaluationSubject
    field: str = Field(min_length=1)
    suggested_value: str | None = None
    rationale: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("suggested_value", "rationale", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class EvaluationFinding(ValidationModel):
    """Evidence-backed issue or observation from an evaluator."""

    code: str = Field(pattern=ISSUE_CODE_PATTERN)
    severity: IssueSeverity = "warn"
    message: str = Field(min_length=1)
    subject: EvaluationSubject | None = None
    evidence_items: tuple[ValidationEvidence, ...] = Field(default_factory=tuple)
    correction: EvaluationCorrection | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_items", mode="before")
    @classmethod
    def _normalize_evidence_items(cls, value: Any) -> tuple[ValidationEvidence, ...]:
        if value is None:
            return ()
        if isinstance(value, ValidationEvidence):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            return tuple(
                item
                if isinstance(item, ValidationEvidence)
                else ValidationEvidence.model_validate(item)
                for item in value
            )
        raise ValueError("evidence_items must be a ValidationEvidence or sequence")

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class EvaluationResult(ValidationModel):
    """Generic no-verdict evaluation result shared by agents and validators."""

    schema_version: Literal["evaluation-signals/v1"] = Field(
        default=EVALUATION_SCHEMA_VERSION
    )
    domain: str = Field(min_length=1)
    status: EvaluationStatus
    findings: tuple[EvaluationFinding, ...] = Field(default_factory=tuple)
    corrections: tuple[EvaluationCorrection, ...] = Field(default_factory=tuple)
    evidence_items: tuple[ValidationEvidence, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("findings", mode="before")
    @classmethod
    def _normalize_findings(cls, value: Any) -> tuple[EvaluationFinding, ...]:
        if value is None:
            return ()
        if isinstance(value, EvaluationFinding):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            return tuple(
                item
                if isinstance(item, EvaluationFinding)
                else EvaluationFinding.model_validate(item)
                for item in value
            )
        raise ValueError("findings must be an EvaluationFinding or sequence")

    @field_validator("corrections", mode="before")
    @classmethod
    def _normalize_corrections(cls, value: Any) -> tuple[EvaluationCorrection, ...]:
        if value is None:
            return ()
        if isinstance(value, EvaluationCorrection):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            return tuple(
                item
                if isinstance(item, EvaluationCorrection)
                else EvaluationCorrection.model_validate(item)
                for item in value
            )
        raise ValueError("corrections must be an EvaluationCorrection or sequence")

    @field_validator("evidence_items", mode="before")
    @classmethod
    def _normalize_evidence_items(cls, value: Any) -> tuple[ValidationEvidence, ...]:
        if value is None:
            return ()
        if isinstance(value, ValidationEvidence):
            return (value,)
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            return tuple(
                item
                if isinstance(item, ValidationEvidence)
                else ValidationEvidence.model_validate(item)
                for item in value
            )
        raise ValueError("evidence_items must be a ValidationEvidence or sequence")

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


def material_self_evaluation_result_from_signals(
    *,
    evaluation_signals: Mapping[str, Any],
    previous_prim_feedback: Mapping[str, Any] | None = None,
    resolved_assignments: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """Convert Material Agent self-evaluation signals into validation-core output."""

    prediction = _as_mapping(evaluation_signals.get("prediction_analysis"))
    visual = _as_mapping(evaluation_signals.get("visual_evaluation"))
    grounding = _as_mapping(evaluation_signals.get("visual_grounding"))
    previous_prim_feedback = previous_prim_feedback or {}
    resolved_assignments = resolved_assignments or {}

    evidence_items = _material_evidence_items(visual=visual, grounding=grounding)
    findings: list[EvaluationFinding] = []
    corrections: list[EvaluationCorrection] = []

    for violation in _mapping_list(prediction.get("symmetry_violations")):
        prim_a = str(violation.get("prim_a", "")).strip()
        prim_b = str(violation.get("prim_b", "")).strip()
        subject_id = " | ".join(part for part in (prim_a, prim_b) if part)
        suggested = _clean_optional_text(violation.get("suggested"))
        correction = None
        if suggested and prim_a:
            correction = EvaluationCorrection(
                subject=EvaluationSubject(kind="prim_pair", identifier=subject_id),
                field="material",
                suggested_value=suggested,
                rationale="Symmetric prims should share a coherent material.",
                metadata={"source": "prediction_analysis"},
            )
            corrections.append(correction)
        findings.append(
            EvaluationFinding(
                code="material.symmetry_mismatch",
                severity="warn",
                message=(
                    f"Symmetric prims have different materials: "
                    f"{prim_a or 'unknown'} vs {prim_b or 'unknown'}."
                ),
                subject=EvaluationSubject(
                    kind="prim_pair",
                    identifier=subject_id or "unknown",
                    metadata={
                        key: _json_value(value) for key, value in violation.items()
                    },
                ),
                correction=correction,
                metadata={"source": "prediction_analysis"},
            )
        )

    for violation in _mapping_list(prediction.get("consistency_violations")):
        group_name = str(violation.get("group_name", "repeated_group")).strip()
        suggested = _clean_optional_text(violation.get("suggested"))
        correction = None
        if suggested:
            correction = EvaluationCorrection(
                subject=EvaluationSubject(kind="prim_group", identifier=group_name),
                field="material",
                suggested_value=suggested,
                rationale="Repeated or visually similar prims should be consistent.",
                metadata={"source": "prediction_analysis"},
            )
            corrections.append(correction)
        findings.append(
            EvaluationFinding(
                code="material.consistency_mismatch",
                severity="warn",
                message=f"Repeated material group is inconsistent: {group_name}.",
                subject=EvaluationSubject(
                    kind="prim_group",
                    identifier=group_name,
                    metadata={
                        key: _json_value(value) for key, value in violation.items()
                    },
                ),
                correction=correction,
                metadata={"source": "prediction_analysis"},
            )
        )

    for issue in _string_list(visual.get("issues")):
        if _is_empty_issue(issue):
            continue
        findings.append(
            EvaluationFinding(
                code="material.visual_issue",
                severity="warn",
                message=issue,
                subject=EvaluationSubject(kind="asset", identifier="current_result"),
                evidence_items=tuple(evidence_items),
                metadata={"source": "visual_evaluation"},
            )
        )

    for correction_record in _mapping_list(visual.get("label_based_corrections")):
        labels = _string_list(
            correction_record.get("target_label_ids")
        ) or _string_list(correction_record.get("label_ids"))
        prim_paths = _string_list(correction_record.get("prim_paths"))
        issue = _clean_optional_text(correction_record.get("issue"))
        suggested = _clean_optional_text(correction_record.get("suggested_material"))
        subject_id = ", ".join(labels or prim_paths) or "unknown"
        correction = None
        if suggested and suggested.lower() != "unknown":
            correction = EvaluationCorrection(
                subject=EvaluationSubject(
                    kind="label_group" if labels else "prim_group",
                    identifier=subject_id,
                    metadata={
                        "label_ids": labels,
                        "prim_paths": prim_paths,
                    },
                ),
                field="material",
                suggested_value=suggested,
                rationale=issue,
                metadata={"source": "visual_evaluation"},
            )
            corrections.append(correction)
        findings.append(
            EvaluationFinding(
                code="material.label_correction",
                severity="warn",
                message=issue or f"Label-grounded material issue for {subject_id}.",
                subject=EvaluationSubject(
                    kind="label_group" if labels else "prim_group",
                    identifier=subject_id,
                    metadata={
                        key: _json_value(value)
                        for key, value in correction_record.items()
                    },
                ),
                evidence_items=tuple(evidence_items),
                correction=correction,
                metadata={"source": "visual_evaluation"},
            )
        )

    for prim_path, feedback in previous_prim_feedback.items():
        prim = str(prim_path).strip()
        message = _clean_optional_text(feedback)
        if not prim or not message:
            continue
        findings.append(
            EvaluationFinding(
                code="material.prim_feedback",
                severity="warn",
                message=message,
                subject=EvaluationSubject(kind="prim", identifier=prim),
                metadata={"source": "previous_prim_feedback"},
            )
        )

    for prim_path, material in resolved_assignments.items():
        prim = str(prim_path).strip()
        suggested = _clean_optional_text(material)
        if not prim or not suggested:
            continue
        corrections.append(
            EvaluationCorrection(
                subject=EvaluationSubject(kind="prim", identifier=prim),
                field="material",
                suggested_value=suggested,
                rationale="Resolved by material self-evaluation evidence.",
                metadata={"source": "resolved_assignments"},
            )
        )

    return EvaluationResult(
        domain="material_agent.material_assignment",
        status=_combined_material_status(prediction=prediction, visual=visual),
        findings=tuple(_dedupe_findings(findings)),
        corrections=tuple(_dedupe_corrections(corrections)),
        evidence_items=tuple(evidence_items),
        metadata={
            "source_schema_version": evaluation_signals.get("schema_version"),
            "prediction_status": prediction.get("status"),
            "visual_status": visual.get("status"),
            "visual_grounding_status": grounding.get("status"),
        },
    )


def _material_evidence_items(
    *,
    visual: Mapping[str, Any],
    grounding: Mapping[str, Any],
) -> list[ValidationEvidence]:
    evidence: list[ValidationEvidence] = []
    for kind, key in (
        ("reference_image", "reference_image_paths"),
        ("current_render", "rendered_image_paths"),
        ("turntable_contact_sheet", "turntable_contact_sheet_image_paths"),
        ("visual_grounding_overlay", "visual_grounding_image_paths"),
    ):
        for path in _string_list(visual.get(key)):
            evidence.append(ValidationEvidence(kind=kind, path=path))
    for kind, key in (
        ("visual_grounding_packet", "packet_path"),
        ("visual_grounding_report", "html_report_path"),
    ):
        path = _clean_optional_text(grounding.get(key))
        if path:
            evidence.append(ValidationEvidence(kind=kind, path=path))
    return _dedupe_evidence(evidence)


def _combined_material_status(
    *,
    prediction: Mapping[str, Any],
    visual: Mapping[str, Any],
) -> EvaluationStatus:
    statuses = {
        _clean_optional_text(prediction.get("status")) or "skipped",
        _clean_optional_text(visual.get("status")) or "skipped",
    }
    if "error" in statuses:
        return "error"
    if statuses == {"completed"}:
        return "completed"
    if "partial" in statuses or "completed" in statuses:
        return "partial"
    return "skipped"


def _dedupe_findings(findings: Sequence[EvaluationFinding]) -> list[EvaluationFinding]:
    seen: set[tuple[str, str, str]] = set()
    result: list[EvaluationFinding] = []
    for finding in findings:
        subject_id = finding.subject.identifier if finding.subject else ""
        key = (finding.code, subject_id, finding.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


def _dedupe_corrections(
    corrections: Sequence[EvaluationCorrection],
) -> list[EvaluationCorrection]:
    seen: set[tuple[str, str, str, str | None]] = set()
    result: list[EvaluationCorrection] = []
    for correction in corrections:
        key = (
            correction.subject.kind,
            correction.subject.identifier,
            correction.field,
            correction.suggested_value,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(correction)
    return result


def _dedupe_evidence(
    evidence: Sequence[ValidationEvidence],
) -> list[ValidationEvidence]:
    seen: set[tuple[str, str | None, str | None]] = set()
    result: list[ValidationEvidence] = []
    for item in evidence:
        key = (item.kind, item.path, item.subject)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str | Path):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [text for item in value if (text := str(item).strip())]
    return [str(value).strip()] if str(value).strip() else []


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_empty_issue(value: str) -> bool:
    normalized = value.strip().lower().rstrip(".")
    return normalized in {"none", "no issues", "no visible issues", "n/a"}
