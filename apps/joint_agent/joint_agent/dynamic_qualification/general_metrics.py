# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General dynamic behavior semantics over already captured trajectory evidence.

This module owns the first trusted :class:`~joint_agent.dynamic_qualification.
verification.DynamicQualificationEvidenceAdapter`. It launches no runtime and
reads no path: it interprets exactly the opaque bytes the verifier captured and
retained, and reduces them to the five general metrics required of every
dynamic profile:

``finite_state``
    The state never leaves its admitted finite envelope. A non-finite sample
    and a magnitude beyond the admitted maximum are the same defect class, so
    they share one stable failure code.
``anchor_separation``
    Attachment/anchor frames never separate beyond their admitted distance.
``locked_dof_drift``
    Forbidden degrees of freedom never move away from their initial value.
``allowed_dof_response``
    Every allowed degree of freedom actually responds while a stimulus bound to
    its subject is applied.
``limit_error``
    Travel never violates its admitted constraint limits.

The forbidden-DOF and allowed-DOF checks are deliberately evaluated over
disjoint coordinate sets and disjoint sample windows so a single deliberately
broken fixture fails exactly one of them.

Tolerances are never adapter-owned defaults: each metric reads its exact bound
from the admitted profile, and a profile that does not declare the complete
five-metric battery with the expected kind, unit, and bound orientation for
every subject is rejected rather than partially interpreted.

Nothing an unearned PASS would depend on is left to the evidence to declare
about itself. The admitted profile — not the trajectory — owns which
coordinates of a subject may move, which are locked, and what unit each is
measured in. Every trajectory names the execution its own attempt attests, so a
measurement replayed from an earlier passing run is refused rather than
counted twice. Provenance behind an inapplicable decision is bound to the
admitted profile, scenario and repetition exactly like provenance behind a
completed one, because skipping the battery is a verdict too.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self, cast

from pydantic import Field, ValidationError, field_validator, model_validator

from joint_agent.dynamic_qualification.contracts import (
    AdapterApplicabilityV1,
    AdapterMetricObservationV1,
    AdapterSemanticsIdentityV1,
    ArtifactIdentityClaimV1,
    AttemptExecutionAttestationV1,
    DynamicQualificationContractModel,
    DynamicQualificationProfileV1,
    MetricSpecV1,
    QualificationResultRecordV1,
    QualificationSubjectV1,
    RuntimeIdentityV1,
    _require_sorted_unique,
    require_canonical_token,
)

if TYPE_CHECKING:  # pragma: no cover - imports only for static type checking
    from joint_agent.dynamic_qualification.verification import CapturedArtifactHandle

ADAPTER_ID = "joint-general-dynamics"
ADAPTER_VERSION = "1.0.0"
PROVENANCE_SCHEMA_VERSION: Literal["joint-general-dynamics-provenance-v1"] = (
    "joint-general-dynamics-provenance-v1"
)
TRAJECTORY_SCHEMA_VERSION: Literal["joint-general-dynamics-trajectory-v1"] = (
    "joint-general-dynamics-trajectory-v1"
)
PROVENANCE_EVIDENCE_ROLE = "run-provenance"
TRAJECTORY_EVIDENCE_ROLE = "subject-trajectory"
PROVENANCE_EVIDENCE_KEY = (PROVENANCE_EVIDENCE_ROLE, "")
DETERMINISM_FAILURE_CODE = "physical_behavior.dynamic_metric_determinism_failed"

# Exactly the semantics hashed into SEMANTICS_SHA256 below. Any change to a
# metric ID, kind, unit, bound orientation, reduction, or failure code must
# change that pin, which retires every profile that named the old semantics.
SEMANTICS_SHA256 = "5b02caaa6468333c3d843d9436f3f5ee13d2844ef84b8b34eb9be86a49faf8c6"

_MAX_SAMPLE_STEP = 1_000_000_000
_DECIMAL_LITERAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,18})?$")


class GeneralMetricsEvidenceError(ValueError):
    """Captured evidence or a profile shape is unsupported by these semantics."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class _GeneralMetricSemantics:
    """One general metric, its exact reduction, and its stable failure code."""

    metric_id: str
    kind: str
    units: tuple[str, ...]
    bound: Literal["maximum", "minimum"]
    reduction: str
    failure_code: str


# Failure codes keep the static lane's `<subject>_<verb>` snake_case spelling
# under the dotted `physical_behavior.` namespace the dynamic lane already
# publishes, so both lanes read the same way in a Validation Agent verdict.
GENERAL_METRIC_SEMANTICS: tuple[_GeneralMetricSemantics, ...] = (
    _GeneralMetricSemantics(
        metric_id="allowed_dof_response",
        kind="general.allowed_dof_response",
        units=("meter", "radian"),
        bound="minimum",
        reduction="min_over_allowed_dof_of_max_abs_delta_from_initial_in_stimulus_window",
        failure_code="physical_behavior.dynamic_allowed_dof_response_absent",
    ),
    _GeneralMetricSemantics(
        metric_id="anchor_separation",
        kind="general.anchor_separation",
        units=("meter",),
        bound="maximum",
        reduction="max_over_samples_of_anchor_separation",
        failure_code="physical_behavior.dynamic_anchor_separation_exceeded",
    ),
    _GeneralMetricSemantics(
        metric_id="finite_state",
        kind="general.finite_state",
        units=("dimensionless",),
        bound="maximum",
        reduction="max_over_samples_of_max_abs_state_or_non_finite",
        failure_code="physical_behavior.dynamic_state_not_finite",
    ),
    _GeneralMetricSemantics(
        metric_id="limit_error",
        kind="general.limit_error",
        units=("meter", "radian"),
        bound="maximum",
        reduction="max_over_samples_of_limit_error",
        failure_code="physical_behavior.dynamic_limit_error_exceeded",
    ),
    _GeneralMetricSemantics(
        metric_id="locked_dof_drift",
        kind="general.locked_dof_drift",
        units=("meter", "radian"),
        bound="maximum",
        reduction="max_over_locked_dof_of_max_abs_delta_from_initial",
        failure_code="physical_behavior.dynamic_locked_dof_drift_exceeded",
    ),
)
_SEMANTICS_BY_METRIC_ID: dict[str, _GeneralMetricSemantics] = {
    item.metric_id: item for item in GENERAL_METRIC_SEMANTICS
}
_REQUIRED_METRIC_IDS = frozenset(_SEMANTICS_BY_METRIC_ID)


def semantics_document() -> dict[str, Any]:
    """Return the exact document pinned by :data:`SEMANTICS_SHA256`."""

    return {
        "adapter_id": ADAPTER_ID,
        "adapter_version": ADAPTER_VERSION,
        "applicability_binding": (
            "inapplicable_provenance_binds_profile_scenario_and_repetition"
        ),
        "determinism_failure_code": DETERMINISM_FAILURE_CODE,
        "dof_partition": "profile_subject_allowed_and_locked_dof",
        "dof_units": "profile_subject_dof_unit_equals_metric_unit",
        "evidence_roles": {
            "provenance": list(PROVENANCE_EVIDENCE_KEY),
            "trajectory_per_subject": TRAJECTORY_EVIDENCE_ROLE,
        },
        "metrics": [
            {
                "bound": item.bound,
                "failure_code": item.failure_code,
                "kind": item.kind,
                "metric_id": item.metric_id,
                "reduction": item.reduction,
                "units": list(item.units),
            }
            for item in GENERAL_METRIC_SEMANTICS
        ],
        "non_finite_policy": "subject_battery_non_finite",
        "response_window": "half_open_stimulus_step_span",
        "sample_grid": "range(0, duration_steps + 1, sample_every_steps)",
        "schema_versions": {
            "provenance": PROVENANCE_SCHEMA_VERSION,
            "trajectory": TRAJECTORY_SCHEMA_VERSION,
        },
        "trajectory_execution_binding": "trajectory_execution_id_equals_provenance",
    }


def _require_decimal_literal(value: str, label: str) -> Decimal:
    if _DECIMAL_LITERAL_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded plain decimal literal")
    parsed = Decimal(value)
    if parsed == 0 and value.startswith("-"):
        raise ValueError(f"{label} must not use a negative zero spelling")
    return parsed


def _require_magnitude_literal(value: str, label: str) -> str:
    if _require_decimal_literal(value, label) < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


class GeneralDynamicsDofCoordinateV1(DynamicQualificationContractModel):
    """One exact degree-of-freedom coordinate at one sampled step."""

    dof_id: str
    value: str

    @field_validator("dof_id")
    @classmethod
    def _valid_dof_id(cls, value: str) -> str:
        return require_canonical_token(value, "dof_id")

    @field_validator("value")
    @classmethod
    def _valid_value(cls, value: str) -> str:
        _require_decimal_literal(value, "dof coordinate")
        return value


class GeneralDynamicsSampleV1(DynamicQualificationContractModel):
    """One sampled step of one subject's trajectory.

    A non-finite sample carries no magnitudes and no coordinates: the runtime
    could not report a usable state, so nothing about it may be reduced.
    """

    step: int = Field(ge=0, le=_MAX_SAMPLE_STEP)
    finite: bool
    max_abs_state: str | None = None
    anchor_separation: str | None = None
    limit_error: str | None = None
    dof: tuple[GeneralDynamicsDofCoordinateV1, ...] = ()

    @field_validator("max_abs_state", "anchor_separation", "limit_error")
    @classmethod
    def _valid_magnitudes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_magnitude_literal(value, "trajectory sample magnitude")

    @field_validator("dof")
    @classmethod
    def _canonical_dof(
        cls,
        value: tuple[GeneralDynamicsDofCoordinateV1, ...],
    ) -> tuple[GeneralDynamicsDofCoordinateV1, ...]:
        _require_sorted_unique(
            tuple(item.dof_id for item in value),
            "trajectory sample dof IDs",
        )
        return value

    @model_validator(mode="after")
    def _finiteness_matches_content(self) -> Self:
        magnitudes = (self.max_abs_state, self.anchor_separation, self.limit_error)
        if self.finite:
            if any(item is None for item in magnitudes) or not self.dof:
                raise ValueError(
                    "finite samples require every magnitude and dof coordinate"
                )
        elif any(item is not None for item in magnitudes) or self.dof:
            raise ValueError(
                "non-finite samples must omit magnitudes and dof coordinates"
            )
        return self


class GeneralDynamicsTrajectoryV1(DynamicQualificationContractModel):
    """Adapter-owned per-subject trajectory evidence for one attempt.

    ``execution_id`` binds the measurement to the execution its own attempt
    attests. The verifier already pins that provenance document to the admitted
    profile, scenario and repetition and rejects a reused execution, so a
    trajectory replayed from an earlier passing run names the wrong execution
    and is refused instead of earning a second repetition for free.
    """

    schema_version: Literal["joint-general-dynamics-trajectory-v1"]
    execution_id: str
    subject_id: str
    allowed_dof: tuple[str, ...]
    locked_dof: tuple[str, ...]
    samples: tuple[GeneralDynamicsSampleV1, ...]

    @field_validator("execution_id")
    @classmethod
    def _valid_execution_id(cls, value: str) -> str:
        return require_canonical_token(value, "trajectory execution_id")

    @field_validator("subject_id")
    @classmethod
    def _valid_subject_id(cls, value: str) -> str:
        return require_canonical_token(value, "trajectory subject_id")

    @field_validator("allowed_dof", "locked_dof")
    @classmethod
    def _canonical_dof_set(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("trajectory degree-of-freedom sets must be non-empty")
        for item in value:
            require_canonical_token(item, "trajectory dof_id")
        _require_sorted_unique(value, "trajectory dof IDs")
        return value

    @model_validator(mode="after")
    def _consistent_samples(self) -> Self:
        if set(self.allowed_dof) & set(self.locked_dof):
            raise ValueError("allowed and locked degrees of freedom must be disjoint")
        if not self.samples:
            raise ValueError("trajectory requires at least one sample")
        _require_sorted_unique(
            tuple(item.step for item in self.samples),
            "trajectory sample steps",
        )
        declared = set(self.allowed_dof) | set(self.locked_dof)
        if any(
            {coordinate.dof_id for coordinate in item.dof} != declared
            for item in self.samples
            if item.finite
        ):
            raise ValueError(
                "finite samples must carry exactly the declared degrees of freedom"
            )
        return self


class GeneralDynamicsProvenanceV1(DynamicQualificationContractModel):
    """Adapter-owned execution provenance claimed by one attempt."""

    schema_version: Literal["joint-general-dynamics-provenance-v1"]
    execution_id: str
    repetition_index: int = Field(ge=0)
    profile_id: str
    profile_sha256: str
    scenario_id: str
    scenario_version: str
    runtime: RuntimeIdentityV1
    applicable: bool
    inapplicable_reason_code: str | None = None

    @field_validator(
        "execution_id",
        "profile_id",
        "scenario_id",
        "scenario_version",
    )
    @classmethod
    def _valid_ids(cls, value: str) -> str:
        return require_canonical_token(value, "provenance identifier")

    @model_validator(mode="after")
    def _reason_matches_decision(self) -> Self:
        if self.applicable and self.inapplicable_reason_code is not None:
            raise ValueError("applicable provenance must not carry a reason code")
        if not self.applicable and self.inapplicable_reason_code is None:
            raise ValueError("inapplicable provenance requires a reason code")
        return self


@dataclass(frozen=True, slots=True)
class _SupportedSubject:
    """One subject resolved against these semantics by the admitted profile."""

    subject_id: str
    allowed_dof: tuple[str, ...]
    locked_dof: tuple[str, ...]
    response_spans: tuple[tuple[int, int], ...]

    def responds_at(self, step: int) -> bool:
        """Return whether one sampled step lies in a stimulus window."""

        return any(start <= step < end for start, end in self.response_spans)


@dataclass(frozen=True, slots=True)
class _SupportedProfile:
    """The exact sample grid and per-subject bindings these semantics accept.

    The grid is held as its count and stride rather than as materialized steps:
    an admitted scenario may declare up to a billion steps sampled every step,
    and enumerating that before any evidence is read would exhaust the process
    instead of producing a verdict.
    """

    sample_count: int
    sample_every_steps: int
    subjects: Mapping[str, _SupportedSubject]


@dataclass(frozen=True, slots=True)
class _SubjectReduction:
    """One subject's reduced general metrics for one attempt."""

    finite: bool
    values: Mapping[str, Decimal]


def _require_supported_profile(
    profile: DynamicQualificationProfileV1,
) -> _SupportedProfile:
    """Accept only a profile whose every subject declares the full battery."""

    if profile.attempt_provenance_evidence.key != PROVENANCE_EVIDENCE_KEY:
        raise GeneralMetricsEvidenceError(
            "profile_provenance_role_unsupported",
            "general dynamics semantics require the "
            f"{PROVENANCE_EVIDENCE_ROLE!r} attempt provenance evidence role",
        )
    scenario = profile.scenario
    stride = scenario.sample_every_steps
    sample_count = scenario.duration_steps // stride + 1
    evidence = {item.key: item for item in profile.evidence_requirements}
    metrics: dict[str, list[MetricSpecV1]] = {
        item.subject_id: [] for item in profile.subjects
    }
    for metric in profile.metrics:
        metrics[metric.subject_id].append(metric)
    subjects: dict[str, _SupportedSubject] = {}
    for subject in profile.subjects:
        subject_id = subject.subject_id
        requirement = evidence.get((TRAJECTORY_EVIDENCE_ROLE, subject_id))
        if requirement is None or not requirement.required_on_completed:
            raise GeneralMetricsEvidenceError(
                "profile_trajectory_evidence_missing",
                "every subject requires a required-on-completed "
                f"{TRAJECTORY_EVIDENCE_ROLE!r} evidence role: {subject_id}",
            )
        subject_metrics = metrics[subject_id]
        if {item.metric_id for item in subject_metrics} != _REQUIRED_METRIC_IDS:
            raise GeneralMetricsEvidenceError(
                "profile_metric_battery_mismatch",
                "every subject requires exactly the general metric battery "
                f"{sorted(_REQUIRED_METRIC_IDS)}: {subject_id}",
            )
        for metric in subject_metrics:
            _require_supported_metric(metric)
        _require_supported_dof_units(
            subject,
            {item.metric_id: item.unit for item in subject_metrics},
        )
        response_spans = tuple(
            (item.start_step, item.end_step)
            for item in scenario.stimuli
            if item.subject_id == subject_id
            # The least sampled step at or after ``start_step`` is the next
            # multiple of the stride; the window samples the subject exactly
            # when that step is still inside the half-open span.
            and -(-item.start_step // stride) * stride < item.end_step
        )
        if not response_spans:
            raise GeneralMetricsEvidenceError(
                "profile_response_window_unsampled",
                "every subject requires a stimulus whose half-open step span "
                f"contains a sampled step: {subject_id}",
            )
        subjects[subject_id] = _SupportedSubject(
            subject_id=subject_id,
            allowed_dof=tuple(item.dof_id for item in subject.allowed_dof),
            locked_dof=tuple(item.dof_id for item in subject.locked_dof),
            response_spans=response_spans,
        )
    return _SupportedProfile(
        sample_count=sample_count,
        sample_every_steps=stride,
        subjects=subjects,
    )


def _require_supported_dof_units(
    subject: QualificationSubjectV1,
    units: Mapping[str, str],
) -> None:
    """Require every trusted coordinate to share its metric's declared unit.

    A metric reduces a whole coordinate set to one number, so a set spanning
    linear and angular coordinates has no unit at all, and a coordinate in a
    unit other than its metric's compares a slide against an angular bound.
    """

    for specs, metric_id in (
        (subject.allowed_dof, "allowed_dof_response"),
        (subject.locked_dof, "locked_dof_drift"),
    ):
        expected = units[metric_id]
        if any(item.unit != expected for item in specs):
            raise GeneralMetricsEvidenceError(
                "profile_dof_unit_mismatch",
                f"every {metric_id} coordinate of {subject.subject_id} must "
                f"declare the metric's own unit {expected!r}",
            )


def _require_supported_metric(metric: MetricSpecV1) -> None:
    semantics = _SEMANTICS_BY_METRIC_ID[metric.metric_id]
    if metric.kind != semantics.kind:
        raise GeneralMetricsEvidenceError(
            "profile_metric_kind_mismatch",
            f"metric {metric.metric_id} must declare kind {semantics.kind}",
        )
    if metric.unit not in semantics.units:
        raise GeneralMetricsEvidenceError(
            "profile_metric_unit_unsupported",
            f"metric {metric.metric_id} must declare one of "
            f"{list(semantics.units)} as its unit",
        )
    bounds = (metric.threshold.maximum, metric.threshold.minimum)
    declared, unrelated = bounds if semantics.bound == "maximum" else bounds[::-1]
    if declared is None or unrelated is not None:
        raise GeneralMetricsEvidenceError(
            "profile_metric_threshold_unsupported",
            f"metric {metric.metric_id} must declare exactly a "
            f"{semantics.bound} tolerance",
        )


class GeneralDynamicMetricsAdapter:
    """Trusted, non-simulating interpreter of general dynamic behavior."""

    @property
    def identity(self) -> AdapterSemanticsIdentityV1:
        return AdapterSemanticsIdentityV1(
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
            semantics_sha256=SEMANTICS_SHA256,
        )

    def inspect_context(
        self,
        *,
        profile: DynamicQualificationProfileV1,
        profile_identity: ArtifactIdentityClaimV1,
        input_artifacts: tuple[CapturedArtifactHandle, ...],
        evidence_artifacts: tuple[CapturedArtifactHandle, ...],
    ) -> AdapterApplicabilityV1:
        """Read the runtime and applicability every attempt actually claimed."""

        _require_supported_profile(profile)
        attempts = tuple(
            (item, _parse_provenance(item))
            for item in evidence_artifacts
            if (item.role, item.subject_id or "") == PROVENANCE_EVIDENCE_KEY
        )
        if not attempts:
            # Every attempt failed before it could claim provenance. The
            # verifier owns that outcome as NOT_RUN, so applicability stays
            # true against the pre-admitted runtime.
            return AdapterApplicabilityV1(runtime=profile.runtime, applicable=True)
        documents = tuple(document for _artifact, document in attempts)
        if len({item.runtime for item in documents}) != 1:
            raise GeneralMetricsEvidenceError(
                "provenance_runtime_inconsistent",
                "attempts claimed more than one runtime identity",
            )
        decisions = {
            (item.applicable, item.inapplicable_reason_code) for item in documents
        }
        if len(decisions) != 1:
            raise GeneralMetricsEvidenceError(
                "provenance_applicability_inconsistent",
                "attempts claimed more than one applicability decision",
            )
        first = documents[0]
        if not first.applicable:
            # An inapplicable decision skips the whole dynamic battery and never
            # reaches ``attest_attempt``, so this is the only place its
            # provenance can be tied to the qualification it excuses.
            for artifact, document in attempts:
                _require_bound_provenance(
                    document,
                    profile=profile,
                    profile_identity=profile_identity,
                    repetition_index=artifact.repetition_index,
                )
        return AdapterApplicabilityV1(
            runtime=first.runtime,
            applicable=first.applicable,
            reason_code=first.inapplicable_reason_code,
        )

    def attest_attempt(
        self,
        *,
        profile: DynamicQualificationProfileV1,
        profile_identity: ArtifactIdentityClaimV1,
        repetition_index: int,
        input_artifacts: tuple[CapturedArtifactHandle, ...],
        evidence_artifacts: tuple[CapturedArtifactHandle, ...],
        provenance_artifact: CapturedArtifactHandle,
    ) -> AttemptExecutionAttestationV1:
        """Return exactly what the attempt's own provenance document claims.

        Nothing here is copied from the admitted profile, so the verifier's
        binding check compares two independent sources and a forged claim
        cannot ride along unnoticed.
        """

        _require_supported_profile(profile)
        document = _parse_provenance(provenance_artifact)
        if document.repetition_index != repetition_index:
            raise GeneralMetricsEvidenceError(
                "provenance_repetition_mismatch",
                "provenance document claims a different repetition than its "
                "captured evidence binding",
            )
        if not document.applicable:
            raise GeneralMetricsEvidenceError(
                "provenance_attempt_inapplicable",
                "a completed attempt cannot claim inapplicable provenance",
            )
        return AttemptExecutionAttestationV1(
            execution_id=document.execution_id,
            repetition_index=repetition_index,
            profile_id=document.profile_id,
            profile_sha256=document.profile_sha256,
            scenario_id=document.scenario_id,
            scenario_version=document.scenario_version,
            runtime=document.runtime,
            provenance_artifact=provenance_artifact.identity,
        )

    def observe_metrics(
        self,
        *,
        profile: DynamicQualificationProfileV1,
        repetition_index: int,
        input_artifacts: tuple[CapturedArtifactHandle, ...],
        evidence_artifacts: tuple[CapturedArtifactHandle, ...],
    ) -> tuple[AdapterMetricObservationV1, ...]:
        """Reduce one attempt's trajectories to the admitted metric battery."""

        supported = _require_supported_profile(profile)
        active = tuple(
            metric
            for metric in profile.metrics
            if repetition_index < metric.repeat_policy.repeat_count
        )
        observations: list[AdapterMetricObservationV1] = []
        # Every trajectory binding is resolved before the attempt's provenance
        # so a receipt that retained no measurement at all still reports the
        # missing evidence rather than a provenance defect.
        artifacts = {
            subject_id: _subject_trajectory_artifact(
                evidence_artifacts,
                subject_id=subject_id,
                repetition_index=repetition_index,
            )
            for subject_id in sorted({metric.subject_id for metric in active})
        }
        provenance = _attempt_provenance_document(
            evidence_artifacts,
            repetition_index=repetition_index,
        )
        for subject_id, artifact in artifacts.items():
            reduction = _reduce_trajectory(
                _parse_trajectory(artifact),
                supported.subjects[subject_id],
                supported=supported,
                execution_id=provenance.execution_id,
            )
            observations.extend(
                AdapterMetricObservationV1(
                    subject_id=subject_id,
                    metric_id=metric.metric_id,
                    repetition_index=repetition_index,
                    finite=reduction.finite,
                    value=reduction.values.get(metric.metric_id),
                )
                for metric in active
                if metric.subject_id == subject_id
            )
        return tuple(observations)


def _subject_trajectory_artifact(
    evidence_artifacts: Sequence[CapturedArtifactHandle],
    *,
    subject_id: str,
    repetition_index: int,
) -> CapturedArtifactHandle:
    artifact = next(
        (
            item
            for item in evidence_artifacts
            if item.role == TRAJECTORY_EVIDENCE_ROLE
            and item.subject_id == subject_id
            and item.repetition_index == repetition_index
        ),
        None,
    )
    if artifact is None:
        raise GeneralMetricsEvidenceError(
            "trajectory_evidence_missing",
            f"attempt {repetition_index} retained no trajectory for {subject_id}",
        )
    return artifact


def _attempt_provenance_document(
    evidence_artifacts: Sequence[CapturedArtifactHandle],
    *,
    repetition_index: int,
) -> GeneralDynamicsProvenanceV1:
    """Return the provenance one attempt retained under its own repetition."""

    artifact = next(
        (
            item
            for item in evidence_artifacts
            if (item.role, item.subject_id or "") == PROVENANCE_EVIDENCE_KEY
            and item.repetition_index == repetition_index
        ),
        None,
    )
    if artifact is None:
        raise GeneralMetricsEvidenceError(
            "provenance_evidence_missing",
            f"attempt {repetition_index} retained no provenance document",
        )
    document = _parse_provenance(artifact)
    if document.repetition_index != repetition_index:
        raise GeneralMetricsEvidenceError(
            "provenance_repetition_mismatch",
            "provenance document claims a different repetition than its "
            "captured evidence binding",
        )
    return document


def _require_bound_provenance(
    document: GeneralDynamicsProvenanceV1,
    *,
    profile: DynamicQualificationProfileV1,
    profile_identity: ArtifactIdentityClaimV1,
    repetition_index: int | None,
) -> None:
    """Tie one provenance document to the qualification it speaks for."""

    if (
        document.profile_id != profile.profile_id
        or document.profile_sha256 != profile_identity.sha256
    ):
        raise GeneralMetricsEvidenceError(
            "provenance_profile_mismatch",
            "provenance document claims a different profile than the admitted "
            "profile bytes",
        )
    if (
        document.scenario_id != profile.scenario.scenario_id
        or document.scenario_version != profile.scenario.scenario_version
    ):
        raise GeneralMetricsEvidenceError(
            "provenance_scenario_mismatch",
            "provenance document claims a different scenario than the admitted profile",
        )
    if document.repetition_index != repetition_index:
        raise GeneralMetricsEvidenceError(
            "provenance_repetition_mismatch",
            "provenance document claims a different repetition than its "
            "captured evidence binding",
        )


def _reduce_trajectory(
    trajectory: GeneralDynamicsTrajectoryV1,
    subject: _SupportedSubject,
    *,
    supported: _SupportedProfile,
    execution_id: str,
) -> _SubjectReduction:
    if trajectory.subject_id != subject.subject_id:
        raise GeneralMetricsEvidenceError(
            "trajectory_subject_mismatch",
            "trajectory document subject disagrees with its evidence binding",
        )
    if trajectory.execution_id != execution_id:
        raise GeneralMetricsEvidenceError(
            "trajectory_execution_mismatch",
            "trajectory document claims a different execution than the "
            "provenance its own attempt attests",
        )
    if (
        trajectory.allowed_dof != subject.allowed_dof
        or trajectory.locked_dof != subject.locked_dof
    ):
        raise GeneralMetricsEvidenceError(
            "trajectory_dof_partition_mismatch",
            "trajectory document classifies degrees of freedom differently "
            "from the admitted profile subject",
        )
    samples = trajectory.samples
    stride = supported.sample_every_steps
    if len(samples) != supported.sample_count or any(
        item.step != index * stride for index, item in enumerate(samples)
    ):
        raise GeneralMetricsEvidenceError(
            "trajectory_sample_grid_mismatch",
            "trajectory samples do not cover the admitted scenario sample grid",
        )
    if any(not item.finite for item in samples):
        # One unusable state makes every reduction over this subject
        # meaningless, so the whole battery reports non-finite rather than
        # inventing a value from the remaining samples.
        return _SubjectReduction(finite=False, values={})
    coordinates = tuple(_dof_coordinates(item) for item in samples)
    baseline = coordinates[0]
    response = tuple(
        index for index, item in enumerate(samples) if subject.responds_at(item.step)
    )
    values = {
        "allowed_dof_response": min(
            max(abs(coordinates[index][dof] - baseline[dof]) for index in response)
            for dof in trajectory.allowed_dof
        ),
        "anchor_separation": max(
            _magnitude(item.anchor_separation) for item in samples
        ),
        "finite_state": max(_magnitude(item.max_abs_state) for item in samples),
        "limit_error": max(_magnitude(item.limit_error) for item in samples),
        "locked_dof_drift": max(
            max(abs(coordinate[dof] - baseline[dof]) for coordinate in coordinates)
            for dof in trajectory.locked_dof
        ),
    }
    return _SubjectReduction(finite=True, values=values)


def _dof_coordinates(sample: GeneralDynamicsSampleV1) -> dict[str, Decimal]:
    return {item.dof_id: Decimal(item.value) for item in sample.dof}


def _magnitude(value: str | None) -> Decimal:
    """Return one magnitude a finite sample is already validated to carry."""

    return Decimal(cast(str, value))


def _parse_provenance(
    artifact: CapturedArtifactHandle,
) -> GeneralDynamicsProvenanceV1:
    label = "attempt provenance document"
    payload = _strict_json_payload(artifact, label)
    try:
        return GeneralDynamicsProvenanceV1.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise GeneralMetricsEvidenceError(
            "provenance_document_invalid",
            f"{label} is invalid: {exc}",
        ) from exc


def _parse_trajectory(
    artifact: CapturedArtifactHandle,
) -> GeneralDynamicsTrajectoryV1:
    label = "subject trajectory document"
    payload = _strict_json_payload(artifact, label)
    try:
        return GeneralDynamicsTrajectoryV1.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise GeneralMetricsEvidenceError(
            "trajectory_document_invalid",
            f"{label} is invalid: {exc}",
        ) from exc


def _strict_json_payload(artifact: CapturedArtifactHandle, label: str) -> bytes:
    payload = b"".join(artifact.captured_file.iter_chunks())
    _require_strict_json(payload, label)
    return payload


class _DuplicateJsonKeyError(ValueError):
    pass


def _require_strict_json(payload: bytes, label: str) -> None:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
            document[key] = value
        return document

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        document = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GeneralMetricsEvidenceError(
            "evidence_json_invalid",
            f"{label} must be strict duplicate-free finite JSON: {exc}",
        ) from exc
    if not isinstance(document, dict):
        raise GeneralMetricsEvidenceError(
            "evidence_json_invalid",
            f"{label} root must be an object",
        )


def general_metric_failure_codes(
    record: QualificationResultRecordV1,
) -> tuple[str, ...]:
    """Return the stable failure codes a verified general-metrics record earned.

    The verifier collapses every threshold violation into one aggregate reason
    code. This is the boundary that names which general metric actually failed
    without letting an adapter-specific code leak into the shared record.
    """

    if record.adapter != GENERAL_DYNAMIC_METRICS_ADAPTER.identity:
        raise GeneralMetricsEvidenceError(
            "record_adapter_mismatch",
            "general metric failure codes require a general-dynamics record",
        )
    codes: set[str] = set()
    for metric in record.metrics:
        semantics = _SEMANTICS_BY_METRIC_ID.get(metric.metric_id)
        if semantics is None or semantics.kind != metric.kind:
            raise GeneralMetricsEvidenceError(
                "record_metric_unknown",
                f"record carries a non-general metric: {metric.metric_id}",
            )
        if not metric.threshold_passed:
            codes.add(semantics.failure_code)
        if not metric.determinism_passed:
            codes.add(DETERMINISM_FAILURE_CODE)
    return tuple(sorted(codes))


GENERAL_DYNAMIC_METRICS_ADAPTER = GeneralDynamicMetricsAdapter()


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "GENERAL_DYNAMIC_METRICS_ADAPTER",
    "GENERAL_METRIC_SEMANTICS",
    "GeneralDynamicMetricsAdapter",
    "GeneralDynamicsDofCoordinateV1",
    "GeneralDynamicsProvenanceV1",
    "GeneralDynamicsSampleV1",
    "GeneralDynamicsTrajectoryV1",
    "GeneralMetricsEvidenceError",
    "PROVENANCE_EVIDENCE_ROLE",
    "SEMANTICS_SHA256",
    "TRAJECTORY_EVIDENCE_ROLE",
    "general_metric_failure_codes",
    "semantics_document",
]
