# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted verification of captured dynamic qualification evidence."""

from __future__ import annotations

import hashlib
import json
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from threading import RLock
from typing import Protocol, cast, runtime_checkable

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError
from world_understanding.utils.captured_artifacts import (
    CapturedArtifactCleanupError,
    CapturedArtifactError,
    CapturedOpaqueArtifactResolver,
    CapturedOpaqueFile,
    OpaqueArtifactRequest,
    capture_resolved_opaque_file,
)

from joint_agent.dynamic_qualification.admission import (
    AdmittedDynamicProfile,
    DynamicProfileAdmissionError,
    _active_admitted_dynamic_profile_authority,
    _AdmittedDynamicProfileAuthority,
    _captured_profile_file,
    _require_admitted_dynamic_profile,
    admit_dynamic_profile,
)
from joint_agent.dynamic_qualification.contracts import (
    _NOT_RUN_ATTEMPT_REASON_BY_STATUS,
    AdapterApplicabilityV1,
    AdapterMetricObservationV1,
    AdapterSemanticsIdentityV1,
    ArtifactIdentityClaimV1,
    ArtifactSource,
    AttemptExecutionAttestationV1,
    AttemptExecutionV1,
    AttemptStatus,
    CapturedArtifactIdentityV1,
    DynamicQualificationProfileV1,
    DynamicQualificationReceiptV1,
    MetricEvaluationV1,
    MetricObservationV1,
    MetricSpecV1,
    QualificationResultRecordV1,
    QualificationStatus,
    ReceiptArtifactClaimV1,
)
from joint_agent.dynamic_qualification.general_metrics import (
    GENERAL_DYNAMIC_METRICS_ADAPTER,
)

_AdapterRegistryKey = tuple[str, str, str]

_ADAPTER_REGISTRY_SEAL = object()
_RESULT_RECORD_ADAPTER = TypeAdapter(QualificationResultRecordV1)


class DynamicQualificationVerificationError(RuntimeError):
    """No trusted qualification result could be produced."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class _FreshProfileAuthority:
    """Fresh packaged authority used for one verification transaction."""

    profile: DynamicQualificationProfileV1
    identity: ArtifactIdentityClaimV1
    registry_sha256: str
    capability_manifest_version: str
    capability_manifest_sha256: str


def _attach_infrastructure_failure_notes(
    target: BaseException,
    failure: BaseException,
) -> None:
    pending = deque((failure,))
    while pending:
        current = pending.popleft()
        if isinstance(current, BaseExceptionGroup):
            pending.extendleft(reversed(current.exceptions))
            continue
        target.add_note(
            f"Verification infrastructure failure: {type(current).__name__}: {current}"
        )
        for note in getattr(current, "__notes__", ()):
            target.add_note(f"Verification infrastructure detail: {note}")


@contextmanager
def _verification_integrity_boundary(
    require_integrity: Callable[[], None],
) -> Iterator[None]:
    """Check integrity without replacing a failure from the guarded operation."""
    try:
        yield
    except BaseException as primary_failure:
        try:
            require_integrity()
        except BaseException as integrity_failure:
            _attach_infrastructure_failure_notes(
                primary_failure,
                integrity_failure,
            )
        raise
    require_integrity()


def _contains_capture_cleanup_failure(failure: BaseException) -> bool:
    pending = [failure]
    while pending:
        current = pending.pop()
        if isinstance(current, CapturedArtifactCleanupError):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return False


@dataclass(frozen=True, slots=True)
class CapturedArtifactHandle:
    """Semantic binding plus the only artifact view exposed to an adapter."""

    source: ArtifactSource
    role: str
    subject_id: str | None
    repetition_index: int | None
    captured_file: CapturedOpaqueFile

    @property
    def identity(self) -> CapturedArtifactIdentityV1:
        return CapturedArtifactIdentityV1(
            source=self.source,
            role=self.role,
            subject_id=self.subject_id,
            repetition_index=self.repetition_index,
            uri=self.captured_file.uri,
            sha256=self.captured_file.sha256,
            size_bytes=self.captured_file.size_bytes,
        )


@runtime_checkable
class DynamicQualificationEvidenceAdapter(Protocol):
    """Trusted, non-simulating interpreter for already captured evidence."""

    @property
    def identity(self) -> AdapterSemanticsIdentityV1: ...

    def inspect_context(
        self,
        *,
        profile: DynamicQualificationProfileV1,
        profile_identity: ArtifactIdentityClaimV1,
        input_artifacts: tuple[CapturedArtifactHandle, ...],
        evidence_artifacts: tuple[CapturedArtifactHandle, ...],
    ) -> AdapterApplicabilityV1:
        """Extract runtime identity and applicability from captured evidence.

        The admitted profile's own digest is supplied because an inapplicable
        decision is a verdict too: it skips the whole dynamic battery without
        ever reaching :meth:`attest_attempt`, so the evidence behind it can only
        be bound to the admitted profile here.
        """

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
        """Derive exact execution provenance from the admitted evidence binding."""

    def observe_metrics(
        self,
        *,
        profile: DynamicQualificationProfileV1,
        repetition_index: int,
        input_artifacts: tuple[CapturedArtifactHandle, ...],
        evidence_artifacts: tuple[CapturedArtifactHandle, ...],
    ) -> tuple[AdapterMetricObservationV1, ...]:
        """Interpret one complete attempt without launching a runtime."""


class _TrustedAdapterRegistry:
    """Sealed internal allowlist of trusted evidence interpretation semantics."""

    _adapters: tuple[
        tuple[_AdapterRegistryKey, DynamicQualificationEvidenceAdapter],
        ...,
    ]
    _seal: object

    __slots__ = ("_adapters", "_seal")

    def __new__(
        cls,
        *_args: object,
        **_kwargs: object,
    ) -> _TrustedAdapterRegistry:
        raise TypeError("trusted adapter registries are created only internally")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("trusted adapter registries are immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("trusted adapter registries are immutable")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("trusted adapter registries cannot be subclassed")


def _create_trusted_adapter_registry(
    adapters: tuple[DynamicQualificationEvidenceAdapter, ...] = (),
) -> _TrustedAdapterRegistry:
    keyed: list[tuple[_AdapterRegistryKey, DynamicQualificationEvidenceAdapter]] = []
    for adapter in adapters:
        identity = _adapter_identity(adapter)
        keyed.append((identity.registry_key, adapter))
    keys = tuple(key for key, _adapter in keyed)
    if len(keys) != len(set(keys)):
        raise DynamicQualificationVerificationError(
            "adapter_registry_duplicate",
            "trusted evidence adapter registry contains duplicate identities",
        )
    registry = object.__new__(_TrustedAdapterRegistry)
    object.__setattr__(registry, "_seal", _ADAPTER_REGISTRY_SEAL)
    object.__setattr__(registry, "_adapters", tuple(keyed))
    return registry


def _load_trusted_adapter_registry() -> _TrustedAdapterRegistry:
    """Load production adapters.

    The general dynamic behavior semantics are the only interpretation the
    0.6 lane admits. Registering them promotes no capability: a profile still
    has to be packaged, bound to a manifest row, executed, and verified before
    any row can carry a dynamic result.
    """

    return _create_trusted_adapter_registry((GENERAL_DYNAMIC_METRICS_ADAPTER,))


class VerifiedDynamicQualification:
    """Identity-only handle for a module-owned verified result snapshot."""

    __slots__ = ("__weakref__",)

    def __new__(
        cls,
        *_args: object,
        **_kwargs: object,
    ) -> VerifiedDynamicQualification:
        raise TypeError(
            "VerifiedDynamicQualification is created only by trusted verification"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("verified dynamic qualification results are immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("verified dynamic qualification results are immutable")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("VerifiedDynamicQualification cannot be subclassed")

    @property
    def record(self) -> QualificationResultRecordV1:
        """Return a fresh strict record detached from module-owned authority."""

        return _require_verified_dynamic_qualification(self)


@dataclass(frozen=True, slots=True)
class _VerifiedDynamicQualificationAuthority:
    """Private canonical bytes bound to one live wrapper identity."""

    handle_ref: weakref.ReferenceType[VerifiedDynamicQualification]
    canonical_record_json: bytes


_VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES: dict[
    int,
    _VerifiedDynamicQualificationAuthority,
] = {}
_VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES_LOCK = RLock()


def verify_dynamic_qualification(
    admitted_profile: AdmittedDynamicProfile,
    receipt: DynamicQualificationReceiptV1,
    *,
    resolver: CapturedOpaqueArtifactResolver,
) -> VerifiedDynamicQualification:
    """Verify an untrusted receipt against one live admitted profile.

    Every admitted input and receipt artifact is retained in the same
    :class:`ExitStack` until all interpretation and integrity checks finish.
    Capture, cleanup, contract, or adapter failures return no trusted object.
    """

    try:
        caller_authority = _active_admitted_dynamic_profile_authority(admitted_profile)
    except DynamicProfileAdmissionError as exc:
        raise DynamicQualificationVerificationError(
            "profile_handle_untrusted",
            "verification requires one live profile admission",
        ) from exc

    try:
        with admit_dynamic_profile(caller_authority.profile_id) as authoritative:
            return _verify_dynamic_qualification_against_fresh_authority(
                admitted_profile,
                caller_authority=caller_authority,
                authoritative=authoritative,
                receipt=receipt,
                resolver=resolver,
            )
    except DynamicQualificationVerificationError:
        raise
    except DynamicProfileAdmissionError as exc:
        raise DynamicQualificationVerificationError(
            "profile_authority_refresh_failed",
            f"packaged profile authority could not be refreshed safely: {exc}",
        ) from exc
    except Exception as exc:
        error = DynamicQualificationVerificationError(
            "verification_infrastructure_failed",
            f"dynamic qualification evidence could not be verified safely: {exc}",
        )
        _attach_infrastructure_failure_notes(error, exc)
        raise error from exc


def _verify_dynamic_qualification_against_fresh_authority(
    admitted_profile: AdmittedDynamicProfile,
    *,
    caller_authority: _AdmittedDynamicProfileAuthority,
    authoritative: AdmittedDynamicProfile,
    receipt: DynamicQualificationReceiptV1,
    resolver: CapturedOpaqueArtifactResolver,
) -> VerifiedDynamicQualification:
    authority = _bind_fresh_profile_authority(
        admitted_profile,
        caller_authority=caller_authority,
        authoritative=authoritative,
    )
    profile = authority.profile
    profile_identity = authority.identity
    trusted_receipt = _strict_receipt(receipt)
    _validate_receipt_identity(profile, trusted_receipt)
    adapter = _select_adapter(profile.adapter)
    claim_plan = _validate_claim_plan(profile, trusted_receipt)

    record: QualificationResultRecordV1 | None = None
    try:
        with ExitStack() as captures:
            input_handles = tuple(
                _capture_artifact(
                    captures,
                    resolver=resolver,
                    source="input",
                    role=item.role,
                    subject_id=item.subject_id,
                    repetition_index=None,
                    uri=item.identity.uri,
                    sha256=item.identity.sha256,
                    size_bytes=item.identity.size_bytes,
                    max_bytes=item.max_bytes,
                )
                for item in profile.input_artifacts
            )
            evidence_handles = tuple(
                _capture_artifact(
                    captures,
                    resolver=resolver,
                    source="evidence",
                    role=claim.role,
                    subject_id=claim.subject_id,
                    repetition_index=repetition_index,
                    uri=claim.identity.uri,
                    sha256=claim.identity.sha256,
                    size_bytes=claim.identity.size_bytes,
                    max_bytes=max_bytes,
                )
                for repetition_index, claim, max_bytes in claim_plan
            )
            _require_all_intact((*input_handles, *evidence_handles))
            expected_requests = tuple(
                (item.captured_file, item.captured_file.request)
                for item in (*input_handles, *evidence_handles)
            )

            def require_integrity() -> None:
                _require_verification_integrity(
                    admitted_profile=admitted_profile,
                    caller_authority=caller_authority,
                    authoritative=authoritative,
                    authority=authority,
                    expected_requests=expected_requests,
                    adapter=adapter,
                    adapter_identity=profile.adapter,
                )

            adapter_context = _inspect_adapter_context(
                adapter,
                profile=profile,
                profile_identity=profile_identity,
                input_artifacts=input_handles,
                evidence_artifacts=evidence_handles,
                require_integrity=require_integrity,
            )
            if adapter_context.runtime != profile.runtime:
                raise DynamicQualificationVerificationError(
                    "adapter_runtime_mismatch",
                    "trusted adapter observed a runtime different from the "
                    "pre-admitted profile",
                )
            attempt_attestations = _attest_completed_attempts(
                adapter=adapter,
                profile=profile,
                profile_identity=profile_identity,
                receipt=trusted_receipt,
                input_handles=input_handles,
                evidence_handles=evidence_handles,
                require_integrity=require_integrity,
            )
            record = _evaluate_receipt(
                authority=authority,
                profile=profile,
                profile_identity=profile_identity,
                receipt=trusted_receipt,
                adapter=adapter,
                adapter_context=adapter_context,
                attempt_attestations=attempt_attestations,
                input_handles=input_handles,
                evidence_handles=evidence_handles,
                require_integrity=require_integrity,
            )
            require_integrity()
    except DynamicQualificationVerificationError:
        raise
    except Exception as exc:
        error = DynamicQualificationVerificationError(
            "verification_infrastructure_failed",
            f"dynamic qualification evidence could not be verified safely: {exc}",
        )
        _attach_infrastructure_failure_notes(error, exc)
        raise error from exc

    _require_fresh_profile_authority_integrity(
        admitted_profile,
        caller_authority=caller_authority,
        authoritative=authoritative,
        expected=authority,
    )
    if record is None:  # pragma: no cover - fail-closed control-flow invariant
        raise DynamicQualificationVerificationError(
            "verification_result_missing",
            "dynamic qualification verification produced no result record",
        )
    return _create_verified_dynamic_qualification(record)


def _authoritative_profile(
    admitted: AdmittedDynamicProfile,
) -> DynamicQualificationProfileV1:
    try:
        profile = admitted.profile
        return DynamicQualificationProfileV1.model_validate(
            profile.model_dump(mode="python", round_trip=True),
            strict=True,
        )
    except (ValidationError, ValueError) as exc:
        raise DynamicQualificationVerificationError(
            "admitted_profile_invalid",
            f"admitted profile failed strict revalidation: {exc}",
        ) from exc


def _bind_fresh_profile_authority(
    admitted_profile: AdmittedDynamicProfile,
    *,
    caller_authority: _AdmittedDynamicProfileAuthority,
    authoritative: AdmittedDynamicProfile,
) -> _FreshProfileAuthority:
    """Bind an exposed handle to a newly admitted packaged authority."""

    try:
        if (
            _active_admitted_dynamic_profile_authority(admitted_profile)
            is not caller_authority
        ):
            raise DynamicProfileAdmissionError(
                "profile_handle_authority_mismatch",
                "caller admission authority changed during verification",
            )
        fresh_admission = _active_admitted_dynamic_profile_authority(authoritative)
        _require_admitted_dynamic_profile(admitted_profile)
        _require_admitted_dynamic_profile(authoritative)
        caller_profile = _authoritative_profile(admitted_profile)
        fresh_profile = _authoritative_profile(authoritative)
        caller_identity = admitted_profile.identity
        fresh_identity = authoritative.identity
        _require_profile_capture_binding(admitted_profile, caller_identity)
        _require_profile_capture_binding(authoritative, fresh_identity)
    except DynamicQualificationVerificationError:
        raise
    except (CapturedArtifactError, DynamicProfileAdmissionError) as exc:
        raise DynamicQualificationVerificationError(
            "profile_authority_mismatch",
            "caller profile handle no longer matches fresh packaged authority",
        ) from exc

    caller_metadata = (
        caller_authority.profile_id,
        caller_authority.identity,
        caller_authority.registry_sha256,
        caller_authority.capability_manifest_version,
        caller_authority.capability_manifest_sha256,
    )
    fresh_metadata = (
        fresh_admission.profile_id,
        fresh_admission.identity,
        fresh_admission.registry_sha256,
        fresh_admission.capability_manifest_version,
        fresh_admission.capability_manifest_sha256,
    )
    fresh_identity_tuple = (
        fresh_identity.uri,
        fresh_identity.sha256,
        fresh_identity.size_bytes,
    )
    if (
        caller_metadata != fresh_metadata
        or caller_authority.identity != fresh_identity_tuple
        or caller_identity != fresh_identity
        or caller_profile != fresh_profile
        or fresh_profile.profile_id != fresh_admission.profile_id
    ):
        raise DynamicQualificationVerificationError(
            "profile_authority_mismatch",
            "caller profile identity, semantics, registry, or manifest authority "
            "disagrees with a fresh packaged admission",
        )

    return _FreshProfileAuthority(
        profile=_detached_profile(fresh_profile),
        identity=_detached_profile_identity(fresh_identity),
        registry_sha256=fresh_admission.registry_sha256,
        capability_manifest_version=fresh_admission.capability_manifest_version,
        capability_manifest_sha256=fresh_admission.capability_manifest_sha256,
    )


def _require_fresh_profile_authority_integrity(
    admitted_profile: AdmittedDynamicProfile,
    *,
    caller_authority: _AdmittedDynamicProfileAuthority,
    authoritative: AdmittedDynamicProfile,
    expected: _FreshProfileAuthority,
) -> None:
    if (
        _bind_fresh_profile_authority(
            admitted_profile,
            caller_authority=caller_authority,
            authoritative=authoritative,
        )
        != expected
    ):
        raise DynamicQualificationVerificationError(
            "profile_authority_drift",
            "fresh packaged profile authority changed during verification",
        )


def _strict_receipt(
    receipt: DynamicQualificationReceiptV1,
) -> DynamicQualificationReceiptV1:
    if not isinstance(receipt, DynamicQualificationReceiptV1):
        raise DynamicQualificationVerificationError(
            "receipt_invalid",
            "verification requires a DynamicQualificationReceiptV1",
        )
    try:
        return DynamicQualificationReceiptV1.model_validate(receipt, strict=True)
    except ValidationError as exc:
        raise DynamicQualificationVerificationError(
            "receipt_invalid",
            f"untrusted receipt failed strict revalidation: {exc}",
        ) from exc


def _validate_receipt_identity(
    profile: DynamicQualificationProfileV1,
    receipt: DynamicQualificationReceiptV1,
) -> None:
    if receipt.profile_id != profile.profile_id:
        raise DynamicQualificationVerificationError(
            "receipt_profile_mismatch",
            "receipt profile ID disagrees with the admitted profile",
        )
    if receipt.runtime != profile.runtime:
        raise DynamicQualificationVerificationError(
            "receipt_runtime_mismatch",
            "receipt runtime identity disagrees with the admitted profile",
        )


def _select_adapter(
    identity: AdapterSemanticsIdentityV1,
) -> DynamicQualificationEvidenceAdapter:
    registry = _load_trusted_adapter_registry()
    if (
        not isinstance(registry, _TrustedAdapterRegistry)
        or getattr(registry, "_seal", None) is not _ADAPTER_REGISTRY_SEAL
    ):
        raise DynamicQualificationVerificationError(
            "adapter_registry_untrusted",
            "dynamic qualification requires the sealed internal adapter registry",
        )
    adapter = next(
        (
            candidate
            for key, candidate in registry._adapters
            if key == identity.registry_key
        ),
        None,
    )
    if adapter is None:
        raise DynamicQualificationVerificationError(
            "adapter_not_admitted",
            "no trusted evidence adapter matches the profile semantics identity",
        )
    _require_adapter_identity(adapter, identity)
    return adapter


def _require_adapter_identity(
    adapter: DynamicQualificationEvidenceAdapter,
    expected: AdapterSemanticsIdentityV1,
) -> None:
    observed_identity = _adapter_identity(adapter)
    if observed_identity != expected:
        raise DynamicQualificationVerificationError(
            "adapter_identity_mismatch",
            "adapter registry key and implementation identity disagree",
        )


def _adapter_identity(
    adapter: DynamicQualificationEvidenceAdapter,
) -> AdapterSemanticsIdentityV1:
    try:
        observed = adapter.identity
        return AdapterSemanticsIdentityV1.model_validate(
            observed.model_dump(mode="python", round_trip=True),
            strict=True,
        )
    except Exception as exc:
        raise DynamicQualificationVerificationError(
            "adapter_identity_invalid",
            f"trusted adapter identity is invalid: {exc}",
        ) from exc


def _require_profile_capture_binding(
    admitted: AdmittedDynamicProfile,
    identity: ArtifactIdentityClaimV1,
) -> None:
    captured = _captured_profile_file(admitted)
    if (
        captured.uri != identity.uri
        or captured.sha256 != identity.sha256
        or captured.size_bytes != identity.size_bytes
    ):
        raise DynamicQualificationVerificationError(
            "profile_capture_binding_mismatch",
            "admitted profile metadata disagrees with its retained exact bytes",
        )


def _require_verification_integrity(
    *,
    admitted_profile: AdmittedDynamicProfile,
    caller_authority: _AdmittedDynamicProfileAuthority,
    authoritative: AdmittedDynamicProfile,
    authority: _FreshProfileAuthority,
    expected_requests: tuple[
        tuple[CapturedOpaqueFile, OpaqueArtifactRequest],
        ...,
    ],
    adapter: DynamicQualificationEvidenceAdapter,
    adapter_identity: AdapterSemanticsIdentityV1,
) -> None:
    try:
        _require_fresh_profile_authority_integrity(
            admitted_profile,
            caller_authority=caller_authority,
            authoritative=authoritative,
            expected=authority,
        )
        for captured, expected in expected_requests:
            captured.require_intact()
            if captured.request != expected:
                raise DynamicQualificationVerificationError(
                    "artifact_capture_metadata_drift",
                    "adapter interaction changed retained artifact identity metadata",
                )
        _require_adapter_identity(adapter, adapter_identity)
    except DynamicQualificationVerificationError:
        raise
    except Exception as exc:
        raise DynamicQualificationVerificationError(
            "verification_integrity_failed",
            f"verification integrity changed during adapter execution: {exc}",
        ) from exc


def _detached_profile(
    profile: DynamicQualificationProfileV1,
) -> DynamicQualificationProfileV1:
    return DynamicQualificationProfileV1.model_validate(
        profile.model_dump(mode="python", round_trip=True),
        strict=True,
    )


def _detached_profile_identity(
    identity: ArtifactIdentityClaimV1,
) -> ArtifactIdentityClaimV1:
    return ArtifactIdentityClaimV1.model_validate(
        identity.model_dump(mode="python", round_trip=True),
        strict=True,
    )


def _detached_handles(
    handles: tuple[CapturedArtifactHandle, ...],
) -> tuple[CapturedArtifactHandle, ...]:
    return tuple(
        CapturedArtifactHandle(
            source=item.source,
            role=item.role,
            subject_id=item.subject_id,
            repetition_index=item.repetition_index,
            captured_file=item.captured_file,
        )
        for item in handles
    )


def _validate_claim_plan(
    profile: DynamicQualificationProfileV1,
    receipt: DynamicQualificationReceiptV1,
) -> tuple[tuple[int, ReceiptArtifactClaimV1, int], ...]:
    requirements = {item.key: item for item in profile.evidence_requirements}
    planned: list[tuple[int, ReceiptArtifactClaimV1, int]] = []
    total_bytes = 0
    for attempt in receipt.attempts:
        if attempt.repetition_index >= profile.required_repeat_count:
            raise DynamicQualificationVerificationError(
                "receipt_repetition_out_of_range",
                "receipt contains a repetition outside the admitted metric policies",
            )
        for claim in attempt.artifacts:
            requirement = requirements.get(claim.key)
            if requirement is None:
                raise DynamicQualificationVerificationError(
                    "receipt_artifact_not_admitted",
                    "receipt contains an artifact role/subject binding absent "
                    "from the admitted profile",
                )
            if claim.identity.size_bytes > requirement.max_bytes:
                raise DynamicQualificationVerificationError(
                    "receipt_artifact_budget_exceeded",
                    "receipt artifact exceeds its independently admitted role budget",
                )
            total_bytes += claim.identity.size_bytes
            planned.append((attempt.repetition_index, claim, requirement.max_bytes))
    if len(planned) > profile.max_receipt_artifacts:
        raise DynamicQualificationVerificationError(
            "receipt_artifact_count_exceeded",
            "receipt artifact count exceeds the admitted profile limit",
        )
    if total_bytes > profile.max_receipt_bytes:
        raise DynamicQualificationVerificationError(
            "receipt_total_budget_exceeded",
            "receipt artifact bytes exceed the admitted profile total budget",
        )
    return tuple(planned)


def _capture_artifact(
    captures: ExitStack,
    *,
    resolver: CapturedOpaqueArtifactResolver,
    source: ArtifactSource,
    role: str,
    subject_id: str | None,
    repetition_index: int | None,
    uri: str,
    sha256: str,
    size_bytes: int,
    max_bytes: int,
) -> CapturedArtifactHandle:
    try:
        request = OpaqueArtifactRequest(
            uri=uri,
            sha256=sha256,
            size_bytes=size_bytes,
            max_bytes=max_bytes,
        )
        captured = captures.enter_context(
            capture_resolved_opaque_file(resolver, request)
        )
    except CapturedArtifactError as exc:
        if isinstance(exc, CapturedArtifactCleanupError):
            raise
        if exc.code == "capture_type_mismatch":
            raise DynamicQualificationVerificationError(
                "artifact_capture_type_mismatch",
                "resolver did not yield an exact CapturedOpaqueFile",
            ) from exc
        if exc.code == "capture_representation_invalid":
            raise DynamicQualificationVerificationError(
                "artifact_capture_request_mismatch",
                "resolver yielded an invalid capture representation or request binding",
            ) from exc
        raise DynamicQualificationVerificationError(
            "artifact_capture_failed",
            f"artifact capture failed for {role}: {exc}",
        ) from exc
    except Exception as exc:
        if _contains_capture_cleanup_failure(exc):
            raise
        raise DynamicQualificationVerificationError(
            "artifact_capture_failed",
            f"artifact capture failed for {role}: {exc}",
        ) from exc
    return CapturedArtifactHandle(
        source=source,
        role=role,
        subject_id=subject_id,
        repetition_index=repetition_index,
        captured_file=captured,
    )


def _require_all_intact(
    artifacts: tuple[CapturedArtifactHandle, ...],
) -> None:
    for artifact in artifacts:
        artifact.captured_file.require_intact()


def _inspect_adapter_context(
    adapter: DynamicQualificationEvidenceAdapter,
    *,
    profile: DynamicQualificationProfileV1,
    profile_identity: ArtifactIdentityClaimV1,
    input_artifacts: tuple[CapturedArtifactHandle, ...],
    evidence_artifacts: tuple[CapturedArtifactHandle, ...],
    require_integrity: Callable[[], None],
) -> AdapterApplicabilityV1:
    with _verification_integrity_boundary(require_integrity):
        try:
            observed = adapter.inspect_context(
                profile=_detached_profile(profile),
                profile_identity=_detached_profile_identity(profile_identity),
                input_artifacts=_detached_handles(input_artifacts),
                evidence_artifacts=_detached_handles(evidence_artifacts),
            )
            return AdapterApplicabilityV1.model_validate(
                observed.model_dump(mode="python", round_trip=True),
                strict=True,
            )
        except DynamicQualificationVerificationError:
            raise
        except Exception as exc:
            raise DynamicQualificationVerificationError(
                "adapter_context_invalid",
                f"trusted adapter context observation is invalid: {exc}",
            ) from exc


def _attest_completed_attempts(
    *,
    adapter: DynamicQualificationEvidenceAdapter,
    profile: DynamicQualificationProfileV1,
    profile_identity: ArtifactIdentityClaimV1,
    receipt: DynamicQualificationReceiptV1,
    input_handles: tuple[CapturedArtifactHandle, ...],
    evidence_handles: tuple[CapturedArtifactHandle, ...],
    require_integrity: Callable[[], None],
) -> dict[int, AttemptExecutionAttestationV1]:
    attestations: dict[int, AttemptExecutionAttestationV1] = {}
    provenance_key = profile.attempt_provenance_evidence.key
    for attempt in receipt.attempts:
        if attempt.status != "COMPLETED":
            continue
        attempt_evidence = tuple(
            item
            for item in evidence_handles
            if item.repetition_index == attempt.repetition_index
        )
        provenance = next(
            (
                item
                for item in attempt_evidence
                if (item.role, item.subject_id or "") == provenance_key
            ),
            None,
        )
        if provenance is None:
            continue
        with _verification_integrity_boundary(require_integrity):
            try:
                observed = adapter.attest_attempt(
                    profile=_detached_profile(profile),
                    profile_identity=_detached_profile_identity(profile_identity),
                    repetition_index=attempt.repetition_index,
                    input_artifacts=_detached_handles(input_handles),
                    evidence_artifacts=_detached_handles(attempt_evidence),
                    provenance_artifact=_detached_handles((provenance,))[0],
                )
                attestation = AttemptExecutionAttestationV1.model_validate(
                    observed.model_dump(mode="python", round_trip=True),
                    strict=True,
                )
            except DynamicQualificationVerificationError:
                raise
            except Exception as exc:
                raise DynamicQualificationVerificationError(
                    "adapter_attestation_invalid",
                    f"trusted adapter attempt attestation is invalid: {exc}",
                ) from exc
        expected_provenance = provenance.identity
        if (
            attestation.repetition_index != attempt.repetition_index
            or attestation.profile_id != profile.profile_id
            or attestation.profile_sha256 != profile_identity.sha256
            or attestation.scenario_id != profile.scenario.scenario_id
            or attestation.scenario_version != profile.scenario.scenario_version
            or attestation.runtime != profile.runtime
            or attestation.provenance_artifact != expected_provenance
        ):
            raise DynamicQualificationVerificationError(
                "adapter_attestation_binding_mismatch",
                "trusted attempt attestation disagrees with the admitted execution "
                "binding",
            )
        attestations[attempt.repetition_index] = attestation
    execution_ids = tuple(item.execution_id for item in attestations.values())
    if len(execution_ids) != len(set(execution_ids)):
        raise DynamicQualificationVerificationError(
            "adapter_execution_id_replayed",
            "completed attempts reused trusted execution provenance",
        )
    return attestations


def _evaluate_receipt(
    *,
    authority: _FreshProfileAuthority,
    profile: DynamicQualificationProfileV1,
    profile_identity: ArtifactIdentityClaimV1,
    receipt: DynamicQualificationReceiptV1,
    adapter: DynamicQualificationEvidenceAdapter,
    adapter_context: AdapterApplicabilityV1,
    attempt_attestations: dict[int, AttemptExecutionAttestationV1],
    input_handles: tuple[CapturedArtifactHandle, ...],
    evidence_handles: tuple[CapturedArtifactHandle, ...],
    require_integrity: Callable[[], None],
) -> QualificationResultRecordV1:
    if not adapter_context.applicable:
        if not profile.applicability.allow_inapplicable:
            raise DynamicQualificationVerificationError(
                "profile_requires_applicability",
                "trusted adapter marked a profile inapplicable although NA is disabled",
            )
        if any(item.status != "INAPPLICABLE" for item in receipt.attempts):
            raise DynamicQualificationVerificationError(
                "receipt_applicability_mismatch",
                "trusted inapplicability disagrees with receipt attempt states",
            )
        return _build_record(
            authority=authority,
            profile=profile,
            profile_identity=profile_identity,
            receipt=receipt,
            status="NA",
            reason_codes=cast(tuple[str, ...], (adapter_context.reason_code,)),
            input_handles=input_handles,
            evidence_handles=evidence_handles,
            attempt_attestations=attempt_attestations,
            metrics=(),
        )

    if any(item.status == "INAPPLICABLE" for item in receipt.attempts):
        raise DynamicQualificationVerificationError(
            "receipt_applicability_mismatch",
            "receipt claimed inapplicability but the trusted adapter did not",
        )

    incomplete_reasons = _incomplete_reasons(profile, receipt)
    if incomplete_reasons:
        return _build_record(
            authority=authority,
            profile=profile,
            profile_identity=profile_identity,
            receipt=receipt,
            status="NOT_RUN",
            reason_codes=incomplete_reasons,
            input_handles=input_handles,
            evidence_handles=evidence_handles,
            attempt_attestations=attempt_attestations,
            metrics=(),
        )

    metric_evaluations = _evaluate_metrics(
        profile=profile,
        receipt=receipt,
        adapter=adapter,
        input_handles=input_handles,
        evidence_handles=evidence_handles,
        require_integrity=require_integrity,
    )
    threshold_failed = any(not item.threshold_passed for item in metric_evaluations)
    determinism_failed = any(not item.determinism_passed for item in metric_evaluations)
    reasons: list[str] = []
    if threshold_failed:
        reasons.append("physical_behavior.dynamic_metric_threshold_failed")
    if determinism_failed:
        reasons.append("physical_behavior.dynamic_metric_determinism_failed")
    return _build_record(
        authority=authority,
        profile=profile,
        profile_identity=profile_identity,
        receipt=receipt,
        status="FAIL" if reasons else "PASS",
        reason_codes=tuple(sorted(reasons)),
        input_handles=input_handles,
        evidence_handles=evidence_handles,
        attempt_attestations=attempt_attestations,
        metrics=metric_evaluations,
    )


def _incomplete_reasons(
    profile: DynamicQualificationProfileV1,
    receipt: DynamicQualificationReceiptV1,
) -> tuple[str, ...]:
    attempts = {item.repetition_index: item for item in receipt.attempts}
    expected_indices = set(range(profile.required_repeat_count))
    reasons: set[str] = set()
    if set(attempts) != expected_indices:
        reasons.add("physical_behavior.dynamic_incomplete_attempts")
    required_keys = {
        item.key for item in profile.evidence_requirements if item.required_on_completed
    }
    for attempt in receipt.attempts:
        reason = _attempt_reason_code(attempt.status)
        if reason is not None:
            reasons.add(reason)
        if attempt.status == "COMPLETED":
            observed_keys = {item.key for item in attempt.artifacts}
            if not required_keys.issubset(observed_keys):
                reasons.add("physical_behavior.dynamic_incomplete_evidence")
    return tuple(sorted(reasons))


def _attempt_reason_code(status: AttemptStatus) -> str | None:
    return _NOT_RUN_ATTEMPT_REASON_BY_STATUS.get(status)


def _evaluate_metrics(
    *,
    profile: DynamicQualificationProfileV1,
    receipt: DynamicQualificationReceiptV1,
    adapter: DynamicQualificationEvidenceAdapter,
    input_handles: tuple[CapturedArtifactHandle, ...],
    evidence_handles: tuple[CapturedArtifactHandle, ...],
    require_integrity: Callable[[], None],
) -> tuple[MetricEvaluationV1, ...]:
    observations: dict[tuple[str, str], list[AdapterMetricObservationV1]] = {
        metric.key: [] for metric in profile.metrics
    }
    for attempt in receipt.attempts:
        attempt_evidence = tuple(
            item
            for item in evidence_handles
            if item.repetition_index == attempt.repetition_index
        )
        expected = {
            metric.key
            for metric in profile.metrics
            if attempt.repetition_index < metric.repeat_policy.repeat_count
        }
        with _verification_integrity_boundary(require_integrity):
            try:
                raw = adapter.observe_metrics(
                    profile=_detached_profile(profile),
                    repetition_index=attempt.repetition_index,
                    input_artifacts=_detached_handles(input_handles),
                    evidence_artifacts=_detached_handles(attempt_evidence),
                )
                validated = tuple(
                    AdapterMetricObservationV1.model_validate(
                        item.model_dump(mode="python", round_trip=True),
                        strict=True,
                    )
                    for item in raw
                )
            except Exception as exc:
                raise DynamicQualificationVerificationError(
                    "adapter_observation_invalid",
                    f"trusted adapter metric observation is invalid: {exc}",
                ) from exc
        keys = tuple(item.key for item in validated)
        if len(keys) != len(set(keys)):
            raise DynamicQualificationVerificationError(
                "adapter_observation_duplicate",
                "trusted adapter returned a duplicate subject/metric observation",
            )
        if set(keys) != expected:
            raise DynamicQualificationVerificationError(
                "adapter_observation_coverage",
                "trusted adapter observations do not exactly cover admitted "
                f"subject/metric keys for repetition {attempt.repetition_index}",
            )
        if any(item.repetition_index != attempt.repetition_index for item in validated):
            raise DynamicQualificationVerificationError(
                "adapter_observation_repetition",
                "trusted adapter observation bound the wrong repetition",
            )
        for item in validated:
            observations[item.key].append(item)

    evaluations = tuple(
        _evaluate_metric(metric, tuple(observations[metric.key]))
        for metric in profile.metrics
    )
    return tuple(sorted(evaluations, key=lambda item: item.key))


def _evaluate_metric(
    metric: MetricSpecV1,
    observations: tuple[AdapterMetricObservationV1, ...],
) -> MetricEvaluationV1:
    expected_indices = tuple(range(metric.repeat_policy.repeat_count))
    ordered = tuple(sorted(observations, key=lambda item: item.repetition_index))
    if tuple(item.repetition_index for item in ordered) != expected_indices:
        raise DynamicQualificationVerificationError(
            "adapter_observation_repeat_coverage",
            f"metric {metric.key} does not cover its exact repetition policy",
        )
    records = tuple(
        MetricObservationV1(
            subject_id=item.subject_id,
            metric_id=item.metric_id,
            repetition_index=item.repetition_index,
            finite=item.finite,
            value=item.value,
            threshold_passed=_threshold_passes(metric, item),
        )
        for item in ordered
    )
    threshold_passed = all(item.threshold_passed for item in records)
    determinism_passed = _determinism_passes(metric, records)
    return MetricEvaluationV1(
        subject_id=metric.subject_id,
        metric_id=metric.metric_id,
        kind=metric.kind,
        unit=metric.unit,
        status=("PASS" if threshold_passed and determinism_passed else "FAIL"),
        threshold_passed=threshold_passed,
        determinism_passed=determinism_passed,
        observations=records,
    )


def _threshold_passes(
    metric: MetricSpecV1,
    observation: AdapterMetricObservationV1,
) -> bool:
    if not observation.finite or observation.value is None:
        return False
    value = Fraction(observation.value)
    minimum = metric.threshold.minimum
    maximum = metric.threshold.maximum
    return not (
        (minimum is not None and value < minimum.as_fraction())
        or (maximum is not None and value > maximum.as_fraction())
    )


def _determinism_passes(
    metric: MetricSpecV1,
    observations: tuple[MetricObservationV1, ...],
) -> bool:
    policy = metric.repeat_policy.determinism
    if policy is None:
        return True
    if any(not item.finite or item.value is None for item in observations):
        return False
    values: list[Fraction] = [
        Fraction(cast(Decimal, item.value)) for item in observations
    ]
    span: Fraction = max(values) - min(values)
    return bool(span <= policy.tolerance.as_fraction())


def _build_record(
    *,
    authority: _FreshProfileAuthority,
    profile: DynamicQualificationProfileV1,
    profile_identity: ArtifactIdentityClaimV1,
    receipt: DynamicQualificationReceiptV1,
    status: QualificationStatus,
    reason_codes: tuple[str, ...],
    input_handles: tuple[CapturedArtifactHandle, ...],
    evidence_handles: tuple[CapturedArtifactHandle, ...],
    attempt_attestations: dict[int, AttemptExecutionAttestationV1],
    metrics: tuple[MetricEvaluationV1, ...],
) -> QualificationResultRecordV1:
    captured_profile_identity = CapturedArtifactIdentityV1(
        source="profile",
        role="dynamic_profile",
        uri=profile_identity.uri,
        sha256=profile_identity.sha256,
        size_bytes=profile_identity.size_bytes,
    )
    identities = tuple(
        sorted(
            (
                captured_profile_identity,
                *(item.identity for item in input_handles),
                *(item.identity for item in evidence_handles),
            ),
            key=lambda item: item.key,
        )
    )
    identity_set_sha256 = _artifact_identity_set_sha256(identities)
    return QualificationResultRecordV1(
        receipt_id=receipt.receipt_id,
        profile_id=profile.profile_id,
        capability_id=profile.capability_id,
        registry_sha256=authority.registry_sha256,
        capability_manifest_version=authority.capability_manifest_version,
        capability_manifest_sha256=authority.capability_manifest_sha256,
        scenario_id=profile.scenario.scenario_id,
        scenario_version=profile.scenario.scenario_version,
        runtime=profile.runtime,
        adapter=profile.adapter,
        attempt_provenance_evidence=profile.attempt_provenance_evidence,
        status=status,
        reason_codes=reason_codes,
        attempts=tuple(
            AttemptExecutionV1(
                repetition_index=item.repetition_index,
                status=item.status,
                attestation=attempt_attestations.get(item.repetition_index),
            )
            for item in receipt.attempts
        ),
        artifacts=identities,
        artifact_identity_set_sha256=identity_set_sha256,
        metrics=metrics,
    )


def _artifact_identity_set_sha256(
    identities: tuple[CapturedArtifactIdentityV1, ...],
) -> str:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in identities],
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _create_verified_dynamic_qualification(
    record: QualificationResultRecordV1,
) -> VerifiedDynamicQualification:
    try:
        validated = _RESULT_RECORD_ADAPTER.validate_python(record, strict=True)
        canonical_record_json = _RESULT_RECORD_ADAPTER.dump_json(
            validated,
            round_trip=True,
        )
        _RESULT_RECORD_ADAPTER.validate_json(canonical_record_json, strict=True)
    except (ValidationError, PydanticSerializationError) as exc:
        raise DynamicQualificationVerificationError(
            "verification_record_invalid",
            f"verifier produced an invalid result record: {exc}",
        ) from exc
    verified = object.__new__(VerifiedDynamicQualification)
    authority_key = id(verified)

    def remove_authority(
        handle_ref: weakref.ReferenceType[VerifiedDynamicQualification],
    ) -> None:
        _remove_verified_dynamic_qualification_authority(
            authority_key,
            handle_ref,
        )

    handle_ref = weakref.ref(verified, remove_authority)
    authority = _VerifiedDynamicQualificationAuthority(
        handle_ref=handle_ref,
        canonical_record_json=canonical_record_json,
    )
    with _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES_LOCK:
        if authority_key in _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES:
            raise DynamicQualificationVerificationError(
                "verified_result_identity_collision",
                "verified result identity collided with one live authority",
            )
        _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES[authority_key] = authority
    return verified


def _remove_verified_dynamic_qualification_authority(
    authority_key: int,
    handle_ref: weakref.ReferenceType[VerifiedDynamicQualification],
) -> None:
    with _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES_LOCK:
        authority = _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES.get(authority_key)
        if authority is not None and authority.handle_ref is handle_ref:
            del _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES[authority_key]


def _require_verified_dynamic_qualification(
    verified: VerifiedDynamicQualification,
) -> QualificationResultRecordV1:
    if not isinstance(verified, VerifiedDynamicQualification):
        raise DynamicQualificationVerificationError(
            "verified_result_untrusted",
            "Validation projection requires a verifier-created result",
        )
    authority_key = id(verified)
    with _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES_LOCK:
        authority = _VERIFIED_DYNAMIC_QUALIFICATION_AUTHORITIES.get(authority_key)
        if authority is None or authority.handle_ref() is not verified:
            raise DynamicQualificationVerificationError(
                "verified_result_untrusted",
                "Validation projection requires a verifier-created result",
            )
        canonical_record_json = authority.canonical_record_json
    try:
        return _RESULT_RECORD_ADAPTER.validate_json(
            canonical_record_json,
            strict=True,
        )
    except ValidationError as exc:
        raise DynamicQualificationVerificationError(
            "verified_result_invalid",
            f"verified result failed strict revalidation: {exc}",
        ) from exc


__all__ = [
    "CapturedArtifactHandle",
    "DynamicQualificationEvidenceAdapter",
    "DynamicQualificationVerificationError",
    "VerifiedDynamicQualification",
    "verify_dynamic_qualification",
]
