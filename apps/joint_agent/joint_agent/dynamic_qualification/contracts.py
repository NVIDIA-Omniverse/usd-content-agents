# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict wire contracts for evidence-backed dynamic joint qualification.

These models describe pre-admitted profiles, untrusted execution receipts, and
serializable verification records. A :class:`QualificationResultRecordV1` is a
record shape only; the trusted runtime object is created exclusively by
``verification.verify_dynamic_qualification``.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

DYNAMIC_PROFILE_SCHEMA_VERSION: Literal["joint-dynamic-profile-v1"] = (
    "joint-dynamic-profile-v1"
)
DYNAMIC_PROFILE_REGISTRY_SCHEMA_VERSION: Literal[
    "joint-dynamic-profile-registry-v1"
] = "joint-dynamic-profile-registry-v1"
DYNAMIC_RECEIPT_SCHEMA_VERSION: Literal["joint-dynamic-receipt-v1"] = (
    "joint-dynamic-receipt-v1"
)
DYNAMIC_RESULT_SCHEMA_VERSION: Literal["joint-dynamic-result-v1"] = (
    "joint-dynamic-result-v1"
)

AttemptStatus = Literal[
    "COMPLETED",
    "NOT_STARTED",
    "CRASHED",
    "TIMED_OUT",
    "CANCELLED",
    "INAPPLICABLE",
]
QualificationStatus = Literal["PASS", "FAIL", "NOT_RUN", "NA"]
MetricStatus = Literal["PASS", "FAIL"]
ArtifactSource = Literal["profile", "input", "evidence"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+~-]*$")
_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
_PROFILE_RESOURCE_PREFIX = "data/dynamic_qualification/profiles/"
_MAX_EXACT_INTEGER = (2**63) - 1
_MAX_RATIONAL_DENOMINATOR = 1_000_000_000
_MAX_PROFILE_REPETITIONS = 1_000
_MAX_RECEIPT_ARTIFACTS = 10_000
_MAX_RECEIPT_BYTES = 1 << 50
_MAX_SCENARIO_STEPS = 1_000_000_000
_NOT_RUN_ATTEMPT_REASON_BY_STATUS: dict[AttemptStatus, str] = {
    "NOT_STARTED": "physical_behavior.dynamic_not_started",
    "CRASHED": "physical_behavior.dynamic_crashed",
    "TIMED_OUT": "physical_behavior.dynamic_timed_out",
    "CANCELLED": "physical_behavior.dynamic_cancelled",
}
_NOT_RUN_AGGREGATE_REASON_CODES = frozenset(
    {
        "physical_behavior.dynamic_incomplete_attempts",
        "physical_behavior.dynamic_incomplete_evidence",
    }
)
_NOT_RUN_REASON_CODES = frozenset(_NOT_RUN_ATTEMPT_REASON_BY_STATUS.values()).union(
    _NOT_RUN_AGGREGATE_REASON_CODES
)


def _validation_field_name(info: ValidationInfo) -> str:
    """Return the concrete field name supplied to a field validator."""

    if info.field_name is None:  # pragma: no cover - Pydantic field invariant
        raise ValueError("field validator requires a concrete field name")
    return info.field_name


class DynamicQualificationContractModel(BaseModel):
    """Base for immutable, fail-closed dynamic qualification documents."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )


class RationalV1(DynamicQualificationContractModel):
    """One canonical exact rational.

    Equivalent spellings are rejected rather than normalized so semantic
    identity never depends on a consumer's fraction-reduction behavior.
    """

    numerator: int = Field(ge=-_MAX_EXACT_INTEGER, le=_MAX_EXACT_INTEGER)
    denominator: int = Field(gt=0, le=_MAX_RATIONAL_DENOMINATOR)

    @model_validator(mode="after")
    def _canonical_fraction(self) -> Self:
        if self.numerator == 0 and self.denominator != 1:
            raise ValueError("zero rational must use denominator 1")
        if math.gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("rational must be in reduced form")
        return self

    def as_fraction(self) -> Fraction:
        """Return the exact standard-library representation."""

        return Fraction(self.numerator, self.denominator)


class ArtifactIdentityClaimV1(DynamicQualificationContractModel):
    """Untrusted identity claim without an admission budget."""

    uri: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("uri")
    @classmethod
    def _valid_uri(cls, value: str) -> str:
        if not value or value != value.strip() or "\x00" in value:
            raise ValueError("artifact uri must be canonical nonblank text")
        return value

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(
                "artifact sha256 must be 64 lowercase hexadecimal characters"
            )
        return value


class RuntimeIdentityV1(DynamicQualificationContractModel):
    """Exact simulator/runtime implementation selected by an admitted profile."""

    runtime_id: str
    runtime_version: str
    implementation_sha256: str

    @field_validator("runtime_id", "runtime_version")
    @classmethod
    def _valid_tokens(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("implementation_sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _canonical_sha256(value, "implementation_sha256")


class AdapterSemanticsIdentityV1(DynamicQualificationContractModel):
    """Identity of the trusted evidence interpretation semantics."""

    adapter_id: str
    adapter_version: str
    semantics_sha256: str

    @field_validator("adapter_id", "adapter_version")
    @classmethod
    def _valid_tokens(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("semantics_sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _canonical_sha256(value, "semantics_sha256")

    @property
    def registry_key(self) -> tuple[str, str, str]:
        """Return the complete adapter registry key."""

        return (self.adapter_id, self.adapter_version, self.semantics_sha256)


class DegreeOfFreedomSpecV1(DynamicQualificationContractModel):
    """One trusted degree of freedom and the unit it is measured in."""

    dof_id: str
    unit: str

    @field_validator("dof_id", "unit")
    @classmethod
    def _valid_tokens(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))


class QualificationSubjectV1(DynamicQualificationContractModel):
    """Stable subject named by metric specifications and stimuli.

    The profile — never the evidence — owns which coordinates of a subject are
    allowed to move and which are locked, so a producer cannot classify a
    coordinate to suit the measurement it wants to report.
    """

    subject_id: str
    description: str
    allowed_dof: tuple[DegreeOfFreedomSpecV1, ...]
    locked_dof: tuple[DegreeOfFreedomSpecV1, ...]

    @field_validator("subject_id")
    @classmethod
    def _valid_subject_id(cls, value: str) -> str:
        return require_canonical_token(value, "subject_id")

    @field_validator("description")
    @classmethod
    def _valid_description(cls, value: str) -> str:
        return _canonical_text(value, "subject description")

    @field_validator("allowed_dof", "locked_dof")
    @classmethod
    def _canonical_dof(
        cls,
        value: tuple[DegreeOfFreedomSpecV1, ...],
        info: ValidationInfo,
    ) -> tuple[DegreeOfFreedomSpecV1, ...]:
        field = _validation_field_name(info)
        if not value:
            raise ValueError(f"subject {field} requires at least one coordinate")
        _require_sorted_unique(
            tuple(item.dof_id for item in value),
            f"subject {field} IDs",
        )
        return value

    @model_validator(mode="after")
    def _disjoint_partition(self) -> Self:
        allowed = {item.dof_id for item in self.allowed_dof}
        locked = {item.dof_id for item in self.locked_dof}
        if allowed & locked:
            raise ValueError(
                "subject allowed and locked degrees of freedom must be disjoint"
            )
        return self


class StimulusParameterV1(DynamicQualificationContractModel):
    """One exact numeric stimulus parameter."""

    name: str
    value: RationalV1
    unit: str

    @field_validator("name", "unit")
    @classmethod
    def _valid_tokens(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))


class StimulusV1(DynamicQualificationContractModel):
    """A stimulus applied on exact integer step boundaries."""

    stimulus_id: str
    subject_id: str
    kind: str
    start_step: int = Field(ge=0, le=_MAX_SCENARIO_STEPS)
    end_step: int = Field(gt=0, le=_MAX_SCENARIO_STEPS)
    parameters: tuple[StimulusParameterV1, ...] = ()

    @field_validator("stimulus_id", "subject_id", "kind")
    @classmethod
    def _valid_tokens(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("parameters")
    @classmethod
    def _canonical_parameters(
        cls,
        value: tuple[StimulusParameterV1, ...],
    ) -> tuple[StimulusParameterV1, ...]:
        _require_sorted_unique(
            tuple(item.name for item in value),
            "stimulus parameter names",
        )
        return value

    @model_validator(mode="after")
    def _valid_interval(self) -> Self:
        if self.start_step >= self.end_step:
            raise ValueError(
                "stimulus must use a non-empty [start_step, end_step) span"
            )
        return self


class ScenarioV1(DynamicQualificationContractModel):
    """Deterministic step-indexed scenario owned by an admitted profile."""

    scenario_id: str
    scenario_version: str
    fixed_timestep_seconds: RationalV1
    duration_steps: int = Field(gt=0, le=_MAX_SCENARIO_STEPS)
    sample_every_steps: int = Field(gt=0, le=_MAX_SCENARIO_STEPS)
    seed: int = Field(ge=0, le=_MAX_EXACT_INTEGER)
    initial_state_id: str
    stimuli: tuple[StimulusV1, ...]

    @field_validator("scenario_id", "scenario_version", "initial_state_id")
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("stimuli")
    @classmethod
    def _canonical_stimuli(
        cls,
        value: tuple[StimulusV1, ...],
    ) -> tuple[StimulusV1, ...]:
        _require_sorted_unique(
            tuple(item.stimulus_id for item in value),
            "stimulus IDs",
        )
        return value

    @model_validator(mode="after")
    def _integer_boundaries(self) -> Self:
        if self.fixed_timestep_seconds.numerator <= 0:
            raise ValueError("fixed_timestep_seconds must be strictly positive")
        if self.sample_every_steps > self.duration_steps:
            raise ValueError("sample_every_steps must not exceed duration_steps")
        if any(item.end_step > self.duration_steps for item in self.stimuli):
            raise ValueError("stimulus end_step must not exceed duration_steps")
        return self


class MetricThresholdV1(DynamicQualificationContractModel):
    """Inclusive exact bounds for one trusted metric observation."""

    minimum: RationalV1 | None = None
    maximum: RationalV1 | None = None

    @model_validator(mode="after")
    def _valid_bounds(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("metric threshold requires minimum and/or maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum.as_fraction() > self.maximum.as_fraction()
        ):
            raise ValueError("metric threshold minimum must not exceed maximum")
        return self


class DeterminismPolicyV1(DynamicQualificationContractModel):
    """Exact maximum span accepted across repeated observations."""

    comparator: Literal["max_span"] = "max_span"
    tolerance: RationalV1

    @model_validator(mode="after")
    def _nonnegative_tolerance(self) -> Self:
        if self.tolerance.numerator < 0:
            raise ValueError("determinism tolerance must be nonnegative")
        return self


class MetricRepeatPolicyV1(DynamicQualificationContractModel):
    """Per-metric repetition and determinism policy."""

    repeat_count: int = Field(gt=0, le=_MAX_PROFILE_REPETITIONS)
    determinism: DeterminismPolicyV1 | None = None

    @model_validator(mode="after")
    def _determinism_matches_repetitions(self) -> Self:
        if self.repeat_count == 1 and self.determinism is not None:
            raise ValueError("single-observation metrics must not claim determinism")
        if self.repeat_count > 1 and self.determinism is None:
            raise ValueError("repeated metrics require a determinism policy")
        return self


class MetricSpecV1(DynamicQualificationContractModel):
    """One metric keyed by subject and metric ID.

    Metric kinds are deliberately repeatable. Semantics are pinned by the
    profile's adapter identity, while the composite key owns threshold and
    repetition policy.
    """

    subject_id: str
    metric_id: str
    kind: str
    unit: str
    threshold: MetricThresholdV1
    repeat_policy: MetricRepeatPolicyV1

    @field_validator("subject_id", "metric_id", "kind", "unit")
    @classmethod
    def _valid_tokens(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @property
    def key(self) -> tuple[str, str]:
        """Return the subject-local metric key."""

        return (self.subject_id, self.metric_id)


class ProfileInputArtifactV1(DynamicQualificationContractModel):
    """Exact immutable input selected by a trusted profile."""

    role: str
    subject_id: str | None = None
    identity: ArtifactIdentityClaimV1
    max_bytes: int = Field(gt=0)

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        return require_canonical_token(value, "input artifact role")

    @field_validator("subject_id")
    @classmethod
    def _valid_subject_id(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else require_canonical_token(value, "input artifact subject_id")
        )

    @model_validator(mode="after")
    def _within_admitted_budget(self) -> Self:
        if self.identity.size_bytes > self.max_bytes:
            raise ValueError("input artifact size exceeds its admitted max_bytes")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.role, self.subject_id or "")


class EvidenceRequirementV1(DynamicQualificationContractModel):
    """Per-attempt evidence role and trusted independent byte budget."""

    role: str
    subject_id: str | None = None
    required_on_completed: bool
    max_bytes: int = Field(gt=0)

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        return require_canonical_token(value, "evidence role")

    @field_validator("subject_id")
    @classmethod
    def _valid_subject_id(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else require_canonical_token(value, "evidence subject_id")
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.role, self.subject_id or "")


class AttemptProvenanceEvidenceV1(DynamicQualificationContractModel):
    """Evidence binding from which a trusted adapter derives run provenance."""

    role: str
    subject_id: str | None = None

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        return require_canonical_token(value, "attempt provenance role")

    @field_validator("subject_id")
    @classmethod
    def _valid_subject_id(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else require_canonical_token(value, "attempt provenance subject_id")
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.role, self.subject_id or "")


class ApplicabilityPolicyV1(DynamicQualificationContractModel):
    """Trusted adapter rule controlling whether ``NA`` is available."""

    rule_id: str
    allow_inapplicable: bool

    @field_validator("rule_id")
    @classmethod
    def _valid_rule_id(cls, value: str) -> str:
        return require_canonical_token(value, "applicability rule_id")


class DynamicQualificationProfileV1(DynamicQualificationContractModel):
    """Immutable runtime profile admitted by ID from packaged exact bytes."""

    schema_version: Literal["joint-dynamic-profile-v1"] = DYNAMIC_PROFILE_SCHEMA_VERSION
    profile_id: str
    capability_id: str
    runtime: RuntimeIdentityV1
    adapter: AdapterSemanticsIdentityV1
    applicability: ApplicabilityPolicyV1
    scenario: ScenarioV1
    subjects: tuple[QualificationSubjectV1, ...]
    input_artifacts: tuple[ProfileInputArtifactV1, ...]
    evidence_requirements: tuple[EvidenceRequirementV1, ...]
    attempt_provenance_evidence: AttemptProvenanceEvidenceV1
    max_receipt_artifacts: int = Field(gt=0, le=_MAX_RECEIPT_ARTIFACTS)
    max_receipt_bytes: int = Field(gt=0, le=_MAX_RECEIPT_BYTES)
    metrics: tuple[MetricSpecV1, ...]

    @field_validator("profile_id", "capability_id")
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("subjects")
    @classmethod
    def _canonical_subjects(
        cls,
        value: tuple[QualificationSubjectV1, ...],
    ) -> tuple[QualificationSubjectV1, ...]:
        if not value:
            raise ValueError("profile requires at least one subject")
        _require_sorted_unique(
            tuple(item.subject_id for item in value),
            "profile subject IDs",
        )
        return value

    @field_validator("input_artifacts")
    @classmethod
    def _canonical_inputs(
        cls,
        value: tuple[ProfileInputArtifactV1, ...],
    ) -> tuple[ProfileInputArtifactV1, ...]:
        if not value:
            raise ValueError("profile requires at least one exact input artifact")
        _require_sorted_unique(
            tuple(item.key for item in value),
            "profile input artifact bindings",
        )
        return value

    @field_validator("evidence_requirements")
    @classmethod
    def _canonical_evidence_requirements(
        cls,
        value: tuple[EvidenceRequirementV1, ...],
    ) -> tuple[EvidenceRequirementV1, ...]:
        if not value or not any(item.required_on_completed for item in value):
            raise ValueError(
                "profile requires at least one completed-attempt evidence role"
            )
        _require_sorted_unique(
            tuple(item.key for item in value),
            "profile evidence requirement bindings",
        )
        return value

    @field_validator("metrics")
    @classmethod
    def _canonical_metrics(
        cls,
        value: tuple[MetricSpecV1, ...],
    ) -> tuple[MetricSpecV1, ...]:
        if not value:
            raise ValueError("profile requires at least one metric")
        _require_sorted_unique(
            tuple(item.key for item in value),
            "profile subject/metric keys",
        )
        return value

    @model_validator(mode="after")
    def _cross_references(self) -> Self:
        subject_ids = {item.subject_id for item in self.subjects}
        referenced_subjects = (
            {item.subject_id for item in self.scenario.stimuli}
            | {
                item.subject_id
                for item in self.input_artifacts
                if item.subject_id is not None
            }
            | {
                item.subject_id
                for item in self.evidence_requirements
                if item.subject_id is not None
            }
            | (
                {self.attempt_provenance_evidence.subject_id}
                if self.attempt_provenance_evidence.subject_id is not None
                else set()
            )
            | {item.subject_id for item in self.metrics}
        )
        unknown = referenced_subjects - subject_ids
        if unknown:
            raise ValueError(
                f"profile references unknown subject IDs: {sorted(unknown)}"
            )
        metric_subjects = {item.subject_id for item in self.metrics}
        missing_metrics = subject_ids - metric_subjects
        if missing_metrics:
            raise ValueError(
                f"profile subjects require at least one metric: {sorted(missing_metrics)}"
            )
        evidence = {item.key: item for item in self.evidence_requirements}
        provenance = evidence.get(self.attempt_provenance_evidence.key)
        if provenance is None or not provenance.required_on_completed:
            raise ValueError(
                "attempt provenance evidence must bind one required-on-completed "
                "evidence requirement"
            )
        required_completed_artifacts = self.required_repeat_count * sum(
            item.required_on_completed for item in self.evidence_requirements
        )
        if self.max_receipt_artifacts < required_completed_artifacts:
            raise ValueError(
                "max_receipt_artifacts cannot represent every required-on-completed "
                "evidence binding across all required attempts: "
                f"requires at least {required_completed_artifacts}"
            )
        return self

    @property
    def required_repeat_count(self) -> int:
        """Return the largest per-metric repetition count."""

        return max(item.repeat_policy.repeat_count for item in self.metrics)


class DynamicProfileRegistryEntryV1(DynamicQualificationContractModel):
    """Trusted packaged binding from a profile ID to exact resource bytes."""

    profile_id: str
    capability_id: str
    resource: str
    identity: ArtifactIdentityClaimV1
    max_bytes: int = Field(gt=0)

    @field_validator("profile_id", "capability_id")
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("resource")
    @classmethod
    def _valid_resource(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value.startswith(_PROFILE_RESOURCE_PREFIX)
            or "\\" in value
            or path.is_absolute()
            or path.as_posix() != value
            or any(part in {".", ".."} for part in path.parts)
            or path.suffix != ".json"
        ):
            raise ValueError(
                "profile resource must be a canonical packaged JSON path under "
                f"{_PROFILE_RESOURCE_PREFIX}"
            )
        return value

    @model_validator(mode="after")
    def _exact_resource_binding(self) -> Self:
        expected_uri = f"pkg://joint_agent/{self.resource}"
        if self.identity.uri != expected_uri:
            raise ValueError("profile identity uri must exactly match its resource")
        if self.identity.size_bytes > self.max_bytes:
            raise ValueError("profile resource size exceeds its admitted max_bytes")
        return self


class DynamicProfileRegistryV1(DynamicQualificationContractModel):
    """Versioned production allowlist of dynamic qualification profiles."""

    schema_version: Literal["joint-dynamic-profile-registry-v1"] = (
        DYNAMIC_PROFILE_REGISTRY_SCHEMA_VERSION
    )
    profiles: tuple[DynamicProfileRegistryEntryV1, ...] = ()

    @field_validator("profiles")
    @classmethod
    def _canonical_profiles(
        cls,
        value: tuple[DynamicProfileRegistryEntryV1, ...],
    ) -> tuple[DynamicProfileRegistryEntryV1, ...]:
        _require_sorted_unique(
            tuple(item.profile_id for item in value),
            "dynamic profile registry IDs",
        )
        resources = tuple(item.resource for item in value)
        if len(resources) != len(set(resources)):
            raise ValueError("dynamic profile resources must be unique")
        return value


class ReceiptArtifactClaimV1(DynamicQualificationContractModel):
    """One per-attempt artifact claimed by an untrusted runner."""

    role: str
    subject_id: str | None = None
    identity: ArtifactIdentityClaimV1

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        return require_canonical_token(value, "receipt artifact role")

    @field_validator("subject_id")
    @classmethod
    def _valid_subject_id(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else require_canonical_token(value, "receipt artifact subject_id")
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.role, self.subject_id or "")


class QualificationAttemptReceiptV1(DynamicQualificationContractModel):
    """Untrusted execution-health receipt for one repetition."""

    repetition_index: int = Field(ge=0)
    status: AttemptStatus
    failure_code: str | None = None
    artifacts: tuple[ReceiptArtifactClaimV1, ...] = Field(
        default=(),
        max_length=_MAX_RECEIPT_ARTIFACTS,
    )

    @field_validator("failure_code")
    @classmethod
    def _valid_failure_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _canonical_reason_code(value, "attempt failure_code")

    @field_validator("artifacts")
    @classmethod
    def _canonical_artifacts(
        cls,
        value: tuple[ReceiptArtifactClaimV1, ...],
    ) -> tuple[ReceiptArtifactClaimV1, ...]:
        _require_sorted_unique(
            tuple(item.key for item in value),
            "attempt artifact bindings",
        )
        return value

    @model_validator(mode="after")
    def _failure_code_matches_status(self) -> Self:
        if self.status == "COMPLETED" and self.failure_code is not None:
            raise ValueError("completed attempts must not carry failure_code")
        if (
            self.status
            in {
                "NOT_STARTED",
                "CRASHED",
                "TIMED_OUT",
                "CANCELLED",
            }
            and self.failure_code is None
        ):
            raise ValueError(f"{self.status} attempts require failure_code")
        if self.status == "INAPPLICABLE" and self.failure_code is not None:
            raise ValueError("inapplicable attempts must not carry failure_code")
        return self


class DynamicQualificationReceiptV1(DynamicQualificationContractModel):
    """Untrusted runner receipt.

    It intentionally has no verdict, metric values, adapter selection, or byte
    budgets. Those are owned by the admitted profile and trusted verifier.
    """

    schema_version: Literal["joint-dynamic-receipt-v1"] = DYNAMIC_RECEIPT_SCHEMA_VERSION
    receipt_id: str
    profile_id: str
    runtime: RuntimeIdentityV1
    attempts: tuple[QualificationAttemptReceiptV1, ...] = Field(
        default=(),
        max_length=_MAX_PROFILE_REPETITIONS,
    )

    @field_validator("receipt_id", "profile_id")
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("attempts")
    @classmethod
    def _canonical_attempts(
        cls,
        value: tuple[QualificationAttemptReceiptV1, ...],
    ) -> tuple[QualificationAttemptReceiptV1, ...]:
        _require_sorted_unique(
            tuple(item.repetition_index for item in value),
            "receipt repetition indices",
        )
        return value


class AdapterApplicabilityV1(DynamicQualificationContractModel):
    """Trusted runtime/applicability observation from captured evidence."""

    runtime: RuntimeIdentityV1
    applicable: bool
    reason_code: str | None = None

    @field_validator("reason_code")
    @classmethod
    def _valid_reason_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _canonical_reason_code(value, "applicability reason_code")

    @model_validator(mode="after")
    def _reason_matches_decision(self) -> Self:
        if self.applicable and self.reason_code is not None:
            raise ValueError("applicable decisions must not carry reason_code")
        if not self.applicable and self.reason_code is None:
            raise ValueError("inapplicable decisions require reason_code")
        return self


class AdapterMetricObservationV1(DynamicQualificationContractModel):
    """Trusted adapter observation before threshold evaluation."""

    subject_id: str
    metric_id: str
    repetition_index: int = Field(ge=0)
    finite: bool
    value: Decimal | None = Field(default=None, allow_inf_nan=False)

    @field_validator("subject_id", "metric_id")
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @model_validator(mode="after")
    def _finite_value(self) -> Self:
        if self.finite:
            if self.value is None or not self.value.is_finite():
                raise ValueError("finite observations require one finite Decimal value")
        elif self.value is not None:
            raise ValueError("non-finite observations must omit value")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject_id, self.metric_id)


class CapturedArtifactIdentityV1(DynamicQualificationContractModel):
    """Exact identity proven while all captures were jointly retained."""

    source: ArtifactSource
    role: str
    subject_id: str | None = None
    repetition_index: int | None = Field(default=None, ge=0)
    uri: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("role")
    @classmethod
    def _valid_role(cls, value: str) -> str:
        return require_canonical_token(value, "captured artifact role")

    @field_validator("subject_id")
    @classmethod
    def _valid_subject_id(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else require_canonical_token(value, "captured artifact subject_id")
        )

    @field_validator("uri")
    @classmethod
    def _valid_uri(cls, value: str) -> str:
        if not value or value != value.strip() or "\x00" in value:
            raise ValueError("captured artifact uri must be canonical nonblank text")
        return value

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return _canonical_sha256(value, "captured artifact sha256")

    @model_validator(mode="after")
    def _source_binding(self) -> Self:
        if self.source in {"profile", "input"} and self.repetition_index is not None:
            raise ValueError("profile/input artifacts must not bind a repetition")
        if self.source == "evidence" and self.repetition_index is None:
            raise ValueError("evidence artifacts must bind a repetition")
        return self

    @property
    def key(self) -> tuple[int, int, str, str]:
        source_rank = {"profile": 0, "input": 1, "evidence": 2}[self.source]
        return (
            source_rank,
            -1 if self.repetition_index is None else self.repetition_index,
            self.role,
            self.subject_id or "",
        )


class AttemptExecutionAttestationV1(DynamicQualificationContractModel):
    """Trusted adapter attestation for one actual completed execution."""

    execution_id: str
    repetition_index: int = Field(ge=0)
    profile_id: str
    profile_sha256: str
    scenario_id: str
    scenario_version: str
    runtime: RuntimeIdentityV1
    provenance_artifact: CapturedArtifactIdentityV1

    @field_validator(
        "execution_id",
        "profile_id",
        "scenario_id",
        "scenario_version",
    )
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator("profile_sha256")
    @classmethod
    def _valid_profile_sha256(cls, value: str) -> str:
        return _canonical_sha256(value, "profile_sha256")

    @model_validator(mode="after")
    def _provenance_matches_attempt(self) -> Self:
        if (
            self.provenance_artifact.source != "evidence"
            or self.provenance_artifact.repetition_index != self.repetition_index
        ):
            raise ValueError(
                "attempt attestation provenance must be evidence for the same repetition"
            )
        return self


class AttemptExecutionV1(DynamicQualificationContractModel):
    repetition_index: int = Field(ge=0)
    status: AttemptStatus
    attestation: AttemptExecutionAttestationV1 | None = None

    @model_validator(mode="after")
    def _attestation_matches_attempt(self) -> Self:
        if self.status != "COMPLETED" and self.attestation is not None:
            raise ValueError("only completed attempts may carry an attestation")
        if (
            self.attestation is not None
            and self.attestation.repetition_index != self.repetition_index
        ):
            raise ValueError("attempt attestation bound the wrong repetition")
        return self


class MetricObservationV1(DynamicQualificationContractModel):
    subject_id: str
    metric_id: str
    repetition_index: int = Field(ge=0)
    finite: bool
    value: Decimal | None = Field(default=None, allow_inf_nan=False)
    threshold_passed: bool

    @field_validator("subject_id", "metric_id")
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @model_validator(mode="after")
    def _value_matches_finiteness(self) -> Self:
        if self.finite:
            if self.value is None or not self.value.is_finite():
                raise ValueError("finite metric record requires a finite Decimal value")
        elif self.value is not None or self.threshold_passed:
            raise ValueError("non-finite metric records must fail without a value")
        return self


class MetricEvaluationV1(DynamicQualificationContractModel):
    subject_id: str
    metric_id: str
    kind: str
    unit: str
    status: MetricStatus
    threshold_passed: bool
    determinism_passed: bool
    observations: tuple[MetricObservationV1, ...]

    @field_validator("subject_id", "metric_id", "kind", "unit")
    @classmethod
    def _valid_tokens(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @model_validator(mode="after")
    def _evaluation_consistency(self) -> Self:
        if not self.observations:
            raise ValueError("metric evaluation requires observations")
        expected_status = (
            "PASS" if self.threshold_passed and self.determinism_passed else "FAIL"
        )
        if self.status != expected_status:
            raise ValueError("metric status disagrees with evaluated checks")
        indices = tuple(item.repetition_index for item in self.observations)
        _require_sorted_unique(indices, "metric observation repetition indices")
        if any(
            item.subject_id != self.subject_id or item.metric_id != self.metric_id
            for item in self.observations
        ):
            raise ValueError("metric observations disagree with their evaluation key")
        if self.threshold_passed != all(
            item.threshold_passed for item in self.observations
        ):
            raise ValueError("metric threshold summary disagrees with observations")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject_id, self.metric_id)


class QualificationResultRecordV1(DynamicQualificationContractModel):
    """Serializable result record produced by the trusted verifier.

    ``NA`` means the trusted adapter found the profile inapplicable: every
    retained attempt is ``INAPPLICABLE`` and no execution is attested.
    ``NOT_RUN`` means qualification remained incomplete. It deliberately
    retains completed-and-attested sibling attempts. A completed attempt may
    lack an attestation only when incomplete evidence explains why trusted
    provenance could not be established. Reason codes must also exactly match
    any non-completed attempt states.
    """

    schema_version: Literal["joint-dynamic-result-v1"] = DYNAMIC_RESULT_SCHEMA_VERSION
    receipt_id: str
    profile_id: str
    capability_id: str
    registry_sha256: str
    capability_manifest_version: str
    capability_manifest_sha256: str
    scenario_id: str
    scenario_version: str
    runtime: RuntimeIdentityV1
    adapter: AdapterSemanticsIdentityV1
    attempt_provenance_evidence: AttemptProvenanceEvidenceV1
    status: QualificationStatus
    reason_codes: tuple[str, ...]
    attempts: tuple[AttemptExecutionV1, ...]
    artifacts: tuple[CapturedArtifactIdentityV1, ...]
    artifact_identity_set_sha256: str
    metrics: tuple[MetricEvaluationV1, ...]

    @field_validator(
        "receipt_id",
        "profile_id",
        "capability_id",
        "capability_manifest_version",
        "scenario_id",
        "scenario_version",
    )
    @classmethod
    def _valid_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_canonical_token(value, _validation_field_name(info))

    @field_validator(
        "registry_sha256",
        "capability_manifest_sha256",
        "artifact_identity_set_sha256",
    )
    @classmethod
    def _valid_hashes(cls, value: str, info: ValidationInfo) -> str:
        return _canonical_sha256(value, _validation_field_name(info))

    @field_validator("reason_codes")
    @classmethod
    def _canonical_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _canonical_reason_code(item, "qualification reason code") for item in value
        )
        _require_sorted_unique(normalized, "qualification reason codes")
        return normalized

    @field_validator("attempts")
    @classmethod
    def _canonical_attempts(
        cls,
        value: tuple[AttemptExecutionV1, ...],
    ) -> tuple[AttemptExecutionV1, ...]:
        _require_sorted_unique(
            tuple(item.repetition_index for item in value),
            "result repetition indices",
        )
        return value

    @field_validator("artifacts")
    @classmethod
    def _canonical_artifacts(
        cls,
        value: tuple[CapturedArtifactIdentityV1, ...],
    ) -> tuple[CapturedArtifactIdentityV1, ...]:
        _require_sorted_unique(
            tuple(item.key for item in value),
            "captured artifact bindings",
        )
        return value

    @field_validator("metrics")
    @classmethod
    def _canonical_metrics(
        cls,
        value: tuple[MetricEvaluationV1, ...],
    ) -> tuple[MetricEvaluationV1, ...]:
        _require_sorted_unique(
            tuple(item.key for item in value),
            "result subject/metric keys",
        )
        return value

    @model_validator(mode="after")
    def _status_consistency(self) -> Self:
        profile_artifacts = tuple(
            artifact for artifact in self.artifacts if artifact.source == "profile"
        )
        if len(profile_artifacts) != 1:
            raise ValueError(
                "verified records require exactly one canonical captured profile"
            )
        profile_artifact = profile_artifacts[0]
        if (
            profile_artifact.role != "dynamic_profile"
            or profile_artifact.subject_id is not None
            or profile_artifact.repetition_index is not None
        ):
            raise ValueError(
                "verified records require exactly one canonical captured profile"
            )
        attestations = tuple(
            item.attestation for item in self.attempts if item.attestation is not None
        )
        execution_ids = tuple(item.execution_id for item in attestations)
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError("completed attempts require unique execution IDs")
        evidence = set(self.artifacts)
        profile_sha256 = profile_artifact.sha256
        for attestation in attestations:
            if (
                attestation.profile_id != self.profile_id
                or attestation.profile_sha256 != profile_sha256
                or attestation.scenario_id != self.scenario_id
                or attestation.scenario_version != self.scenario_version
                or attestation.runtime != self.runtime
            ):
                raise ValueError(
                    "attempt attestation disagrees with result profile, scenario, "
                    "or runtime"
                )
            if (
                attestation.provenance_artifact not in evidence
                or (
                    attestation.provenance_artifact.role,
                    attestation.provenance_artifact.subject_id or "",
                )
                != self.attempt_provenance_evidence.key
            ):
                raise ValueError(
                    "attempt attestation does not bind admitted provenance evidence"
                )
        if self.status == "PASS":
            if self.reason_codes or not self.metrics:
                raise ValueError("PASS requires metrics and no reason codes")
            if not self.attempts or any(
                item.status != "COMPLETED" or item.attestation is None
                for item in self.attempts
            ):
                raise ValueError("PASS requires trusted provenance for every attempt")
            if any(item.status != "PASS" for item in self.metrics):
                raise ValueError("PASS cannot contain failing metrics")
        elif self.status == "FAIL":
            if not self.reason_codes or not self.metrics:
                raise ValueError("FAIL requires metrics and reason codes")
            if not self.attempts or any(
                item.status != "COMPLETED" or item.attestation is None
                for item in self.attempts
            ):
                raise ValueError("FAIL requires trusted provenance for every attempt")
            if all(item.status == "PASS" for item in self.metrics):
                raise ValueError("FAIL requires at least one failing metric")
        elif self.status == "NOT_RUN":
            if not self.reason_codes or self.metrics:
                raise ValueError("NOT_RUN requires reason codes and no metric verdicts")
            if any(item.status == "INAPPLICABLE" for item in self.attempts):
                raise ValueError("NOT_RUN cannot contain inapplicable attempts")
            reason_codes = set(self.reason_codes)
            if not reason_codes.issubset(_NOT_RUN_REASON_CODES):
                raise ValueError(
                    "NOT_RUN reason codes must describe incomplete dynamic "
                    "qualification"
                )
            expected_attempt_reasons = {
                _NOT_RUN_ATTEMPT_REASON_BY_STATUS[item.status]
                for item in self.attempts
                if item.status in _NOT_RUN_ATTEMPT_REASON_BY_STATUS
            }
            observed_attempt_reasons = reason_codes.difference(
                _NOT_RUN_AGGREGATE_REASON_CODES
            )
            if observed_attempt_reasons != expected_attempt_reasons:
                raise ValueError(
                    "NOT_RUN reason codes disagree with non-completed attempt states"
                )
            if (
                not self.attempts
                and "physical_behavior.dynamic_incomplete_attempts" not in reason_codes
            ):
                raise ValueError(
                    "NOT_RUN without attempts requires an incomplete attempts reason"
                )
            if (
                "physical_behavior.dynamic_incomplete_evidence" in reason_codes
                and not any(item.status == "COMPLETED" for item in self.attempts)
            ):
                raise ValueError(
                    "NOT_RUN incomplete evidence requires a completed attempt"
                )
            if (
                any(
                    item.status == "COMPLETED" and item.attestation is None
                    for item in self.attempts
                )
                and "physical_behavior.dynamic_incomplete_evidence" not in reason_codes
            ):
                raise ValueError(
                    "NOT_RUN unattested completed attempts require an incomplete "
                    "evidence reason"
                )
            if not expected_attempt_reasons and not reason_codes.intersection(
                _NOT_RUN_AGGREGATE_REASON_CODES
            ):
                raise ValueError(
                    "NOT_RUN completed attempts require an incomplete receipt or "
                    "evidence reason"
                )
        elif self.status == "NA":
            if not self.reason_codes or self.metrics:
                raise ValueError("NA requires reason codes and no metric verdicts")
            if any(
                item.status != "INAPPLICABLE" or item.attestation is not None
                for item in self.attempts
            ):
                raise ValueError(
                    "NA requires only inapplicable attempts without attestations"
                )
        return self


def require_canonical_token(value: str, label: str) -> str:
    """Return one token only when it matches the shared wire-contract grammar."""

    if not value or value != value.strip() or _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be canonical nonblank token text")
    return value


def _canonical_text(value: str, label: str) -> str:
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be canonical nonblank text")
    return value


def _canonical_sha256(value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_reason_code(value: str, label: str) -> str:
    if _REASON_CODE_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a dotted lowercase identifier")
    return value


def _require_sorted_unique(values: tuple[Any, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    if values != tuple(sorted(values)):
        raise ValueError(f"{label} must be sorted canonically")


__all__ = [
    "AdapterApplicabilityV1",
    "AdapterMetricObservationV1",
    "AdapterSemanticsIdentityV1",
    "ArtifactIdentityClaimV1",
    "AttemptExecutionAttestationV1",
    "AttemptExecutionV1",
    "AttemptProvenanceEvidenceV1",
    "AttemptStatus",
    "CapturedArtifactIdentityV1",
    "DegreeOfFreedomSpecV1",
    "DeterminismPolicyV1",
    "DynamicProfileRegistryEntryV1",
    "DynamicProfileRegistryV1",
    "DynamicQualificationContractModel",
    "DynamicQualificationProfileV1",
    "DynamicQualificationReceiptV1",
    "EvidenceRequirementV1",
    "MetricEvaluationV1",
    "MetricObservationV1",
    "MetricRepeatPolicyV1",
    "MetricSpecV1",
    "MetricThresholdV1",
    "ProfileInputArtifactV1",
    "QualificationAttemptReceiptV1",
    "QualificationResultRecordV1",
    "QualificationStatus",
    "QualificationSubjectV1",
    "RationalV1",
    "ReceiptArtifactClaimV1",
    "RuntimeIdentityV1",
    "ScenarioV1",
    "StimulusParameterV1",
    "StimulusV1",
    "require_canonical_token",
]
