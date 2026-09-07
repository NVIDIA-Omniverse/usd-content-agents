# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure projection of verified dynamics into Validation Agent V1."""

from __future__ import annotations

import json
from typing import Any

from world_understanding.validation.models import (
    TemplateStatus,
    ValidationEvidence,
    ValidationIssue,
    ValidationTemplateResult,
)

from joint_agent.dynamic_qualification.contracts import (
    MetricEvaluationV1,
    QualificationResultRecordV1,
    QualificationStatus,
)
from joint_agent.dynamic_qualification.verification import VerifiedDynamicQualification

_TEMPLATE_STATUS_BY_QUALIFICATION: dict[QualificationStatus, TemplateStatus] = {
    "PASS": "passed",
    "FAIL": "failed",
    "NOT_RUN": "error",
    "NA": "skipped",
}


def to_validation_template_result(
    verified: VerifiedDynamicQualification,
) -> ValidationTemplateResult:
    """Project a trusted result without resolving evidence or running simulation."""

    record = verified.record
    return ValidationTemplateResult(
        template_name="physical_behavior",
        status=_validation_status(record.status),
        issues=_issues(record),
        metrics={
            _metric_projection_key(metric.subject_id, metric.metric_id): (
                _metric_payload(metric)
            )
            for metric in record.metrics
        },
        evidence={
            "artifact_identity_set_sha256": record.artifact_identity_set_sha256,
            "artifact_count": len(record.artifacts),
        },
        evidence_items=tuple(
            ValidationEvidence(
                kind="dynamic_qualification_artifact",
                path=None,
                subject=item.subject_id,
                summary=f"{item.source}:{item.role}",
                metadata={
                    "source": item.source,
                    "role": item.role,
                    "repetition_index": item.repetition_index,
                    "uri": item.uri,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                },
            )
            for item in record.artifacts
        ),
        metadata={
            "dynamic_qualification_status": record.status,
            "reason_codes": list(record.reason_codes),
            "receipt_id": record.receipt_id,
            "profile_id": record.profile_id,
            "capability_id": record.capability_id,
            "registry_sha256": record.registry_sha256,
            "capability_manifest_version": record.capability_manifest_version,
            "capability_manifest_sha256": record.capability_manifest_sha256,
            "runtime": record.runtime.model_dump(mode="json"),
            "adapter": record.adapter.model_dump(mode="json"),
            "scenario_id": record.scenario_id,
            "scenario_version": record.scenario_version,
            "attempt_attestations": [
                (
                    None
                    if item.attestation is None
                    else item.attestation.model_dump(mode="json")
                )
                for item in record.attempts
            ],
        },
    )


def _validation_status(status: QualificationStatus) -> TemplateStatus:
    return _TEMPLATE_STATUS_BY_QUALIFICATION[status]


def _issues(
    record: QualificationResultRecordV1,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if record.status == "FAIL":
        for metric in record.metrics:
            if not metric.threshold_passed:
                issues.append(
                    ValidationIssue(
                        code="physical_behavior.dynamic_metric_threshold_failed",
                        severity="fail",
                        message=(
                            "Dynamic metric violated its admitted behavior threshold."
                        ),
                        template_name="physical_behavior",
                        subject=metric.subject_id,
                        details={
                            "metric_id": metric.metric_id,
                            "kind": metric.kind,
                        },
                    )
                )
            if not metric.determinism_passed:
                issues.append(
                    ValidationIssue(
                        code="physical_behavior.dynamic_metric_determinism_failed",
                        severity="fail",
                        message=("Dynamic metric violated its admitted repeat policy."),
                        template_name="physical_behavior",
                        subject=metric.subject_id,
                        details={
                            "metric_id": metric.metric_id,
                            "kind": metric.kind,
                        },
                    )
                )
    elif record.status == "NOT_RUN":
        issues.append(
            ValidationIssue(
                code="physical_behavior.dynamic_not_run",
                severity="fail",
                message=(
                    "Dynamic qualification did not complete; no behavioral "
                    "PASS or FAIL verdict was inferred."
                ),
                template_name="physical_behavior",
                details={"reason_codes": list(record.reason_codes)},
            )
        )
    return tuple(issues)


def _metric_payload(metric: MetricEvaluationV1) -> dict[str, Any]:
    return {
        "kind": metric.kind,
        "unit": metric.unit,
        "status": metric.status,
        "threshold_passed": metric.threshold_passed,
        "determinism_passed": metric.determinism_passed,
        "observations": [
            {
                "repetition_index": item.repetition_index,
                "finite": item.finite,
                "value": None if item.value is None else str(item.value),
                "threshold_passed": item.threshold_passed,
            }
            for item in metric.observations
        ],
    }


def _metric_projection_key(subject_id: str, metric_id: str) -> str:
    """Encode the composite metric identity without delimiter collisions."""

    return json.dumps(
        [subject_id, metric_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


__all__ = ["to_validation_template_result"]
