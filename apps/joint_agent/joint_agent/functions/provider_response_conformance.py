# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal provider-response evidence and conformance gates for Joint Stage 1.

This module deliberately does not define a new public result carrier.  It keeps
bounded request evidence beside the existing Stage 1 and articulation terminal
contracts so provider-backed fixed-pipeline and embedded callers share one decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import weakref
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from threading import Lock, local
from typing import Any, Literal, NoReturn, cast, get_args
from uuid import uuid4

from pydantic import ValidationError
from world_understanding.utils.credentials import redact_sensitive_config

from joint_agent.functions.stage1_schema import (
    has_parseable_stage1_source_response,
    normalize_stage1_prediction_payload,
)

_MAX_RETAINED_RESPONSE_BYTES = 4096
_FAILURE_RECEIPTS_ATTRIBUTE = "_joint_agent_provider_attempt_receipts_v1"
PROVIDER_RESPONSE_CHECKPOINT_CALLBACK_KEY = (
    "provider_response_diagnostics_checkpoint_callback"
)
_STRUCTURE_GENERIC_TOKENS = {
    "a",
    "an",
    "and",
    "articulated",
    "asset",
    "body",
    "equipment",
    "furniture",
    "machine",
    "manipulator",
    "medical",
    "model",
    "object",
    "robot",
    "robotic",
    "system",
    "the",
}
_UNACCEPTABLE_TAXONOMY = {"", "none", "null", "unknown", "unresolved"}

ProviderRequestKind = Literal["initial", "transport_retry", "contract_correction"]
ProviderAttemptOutcome = Literal[
    "response_received",
    "empty_response",
    "transport_error",
    "contract_rejected",
    "accepted",
]


def project_provider_attempt_persistence(
    context: MutableMapping[str, Any],
    *,
    path: Path,
    digest: str,
    path_key: str,
    digest_key: str,
) -> None:
    """Project one journal write and advance an optional outer checkpoint."""
    context[path_key] = str(path)
    context[digest_key] = digest
    checkpoint_callback = context.get(PROVIDER_RESPONSE_CHECKPOINT_CALLBACK_KEY)
    if checkpoint_callback is None:
        return
    if not callable(checkpoint_callback):
        raise TypeError("provider response checkpoint callback must be callable")
    checkpoint_callback(path, digest)


_PROVIDER_REQUEST_KINDS = frozenset(get_args(ProviderRequestKind))
_PROVIDER_ATTEMPT_OUTCOMES = frozenset(get_args(ProviderAttemptOutcome))
WholeAssetStructureReason = Literal[
    "missing_whole_asset_source_contract",
    "zero_whole_asset_assignments",
    "incompatible_whole_asset_taxonomy",
]
_WHOLE_ASSET_STRUCTURE_REASONS = frozenset(get_args(WholeAssetStructureReason))
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PROVIDER_ATTEMPT_JOURNAL_FIELDS = {"attempts", "whole_asset_structure"}


@dataclass(frozen=True)
class RawProviderResponseEvidence:
    """Digest-bound, bounded, redacted evidence for one raw response.

    ``response_sha256`` commits to the complete pre-redaction provider bytes.
    When evidence is redacted or truncated those bytes are intentionally not
    retained, so only ``retained_response_sha256`` is locally recomputable.
    """

    response_sha256: str
    response_bytes: int
    retained_response: str
    retained_response_sha256: str
    retained_bytes: int
    truncated: bool
    redacted: bool


class ProviderAttemptRecorderError(RuntimeError):
    """Typed local failure from the shared attempt recorder boundary."""

    def __init__(
        self,
        message: str,
        *,
        persistence_error: BaseException | None = None,
    ) -> None:
        self.persistence_error = persistence_error
        super().__init__(message)


class _ProviderAttemptDigestMismatchError(ValueError):
    """Internal typed signal for a checkpoint-bound journal identity change."""


def invoke_provider_attempt_recorder[RecorderResult](
    record: Callable[[], RecorderResult],
) -> RecorderResult:
    """Run one thin attempt recorder without exposing persistence as transport."""
    try:
        return record()
    except (ProviderAttemptRecorderError, _ProviderAttemptReceiptCarrierError):
        raise
    except Exception as error:
        raise ProviderAttemptRecorderError(
            f"provider attempt callback failed: {type(error).__name__}"
        ) from error


def _explicit_error_chain(source_error: BaseException) -> tuple[BaseException, ...]:
    """Return one cycle-bounded explicit cause chain without implicit context."""
    error_chain: list[BaseException] = []
    pending = [source_error]
    seen: set[int] = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        error_chain.append(error)
        if error.__cause__ is not None:
            pending.append(error.__cause__)
    return tuple(error_chain)


def provider_attempt_persistence_error(
    source_error: BaseException,
) -> BaseException | None:
    """Return the first explicit recorder failure without following context."""
    for chained_error in _explicit_error_chain(source_error):
        if isinstance(chained_error, ProviderAttemptRecorderError):
            return chained_error.persistence_error or chained_error
    return None


def _try_add_exception_note(source_error: BaseException, note: str) -> None:
    """Attach diagnostic context without trusting a provider exception object."""
    try:
        source_error.add_note(note)
    except Exception:
        pass


def _validated_raw_provider_response_evidence(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> RawProviderResponseEvidence:
    """Rehydrate bounded evidence only after recomputing retained identity."""
    try:
        evidence = RawProviderResponseEvidence(**payload)
    except TypeError as error:
        raise ValueError(f"{label} is malformed") from error
    if asdict(evidence) != payload:
        raise ValueError(f"{label} is malformed")
    if not (
        isinstance(evidence.response_sha256, str)
        and _SHA256_PATTERN.fullmatch(evidence.response_sha256)
        and isinstance(evidence.response_bytes, int)
        and not isinstance(evidence.response_bytes, bool)
        and evidence.response_bytes >= 0
        and isinstance(evidence.retained_response, str)
        and isinstance(evidence.retained_response_sha256, str)
        and _SHA256_PATTERN.fullmatch(evidence.retained_response_sha256)
        and isinstance(evidence.retained_bytes, int)
        and not isinstance(evidence.retained_bytes, bool)
        and isinstance(evidence.truncated, bool)
        and isinstance(evidence.redacted, bool)
    ):
        raise ValueError(f"{label} is malformed")
    retained = evidence.retained_response.encode("utf-8")
    if not (
        evidence.retained_bytes == len(retained)
        and evidence.retained_bytes <= _MAX_RETAINED_RESPONSE_BYTES
        and evidence.retained_response_sha256 == hashlib.sha256(retained).hexdigest()
    ):
        raise ValueError(f"{label} is malformed")
    if (
        not evidence.redacted
        and not evidence.truncated
        and not (
            evidence.response_bytes == evidence.retained_bytes
            and evidence.response_sha256 == evidence.retained_response_sha256
        )
    ):
        raise ValueError(f"{label} is malformed")
    return evidence


@dataclass(frozen=True)
class ProviderAttemptDiagnostic:
    """One bounded transport or contract attempt outcome."""

    request_kind: ProviderRequestKind
    outcome: ProviderAttemptOutcome
    attempt_number: int
    raw_response: RawProviderResponseEvidence | None = None
    error_type: str | None = None
    normalized_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROVIDER_ATTEMPT_FIELDS = frozenset(
    field.name for field in dataclass_fields(ProviderAttemptDiagnostic)
)


@dataclass(frozen=True)
class ProviderAttemptReceipt:
    """In-process identity for one exact journaled provider attempt."""

    sequence_number: int
    entry_id: str | None = None


@dataclass(frozen=True)
class ProviderAttemptRecording:
    """One receipt plus an explicit carrier for an opaque provider failure."""

    receipt: ProviderAttemptReceipt
    failure_carrier: BaseException | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def sequence_number(self) -> int:
        return self.receipt.sequence_number

    @property
    def entry_id(self) -> str | None:
        return self.receipt.entry_id


class _ProviderAttemptReceiptCarrierError(RuntimeError):
    """Carry one exact receipt when a provider exception cannot hold identity."""

    def __init__(
        self,
        namespace: object,
        receipt: ProviderAttemptReceipt,
    ) -> None:
        self.namespace = namespace
        self.receipt = receipt
        self.consumed = False
        super().__init__("provider failure requires an external attempt receipt")


@dataclass(frozen=True)
class ProviderAttemptJournalSnapshot:
    """Immutable projection of ordered provider attempts and structure state."""

    attempts: tuple[Mapping[str, Any], ...]
    whole_asset_structure: WholeAssetStructureEvaluation | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "attempts": [dict(attempt) for attempt in self.attempts],
        }
        if self.whole_asset_structure is not None:
            payload["whole_asset_structure"] = self.whole_asset_structure.to_dict()
        return payload


@dataclass(frozen=True)
class ProviderAttemptPersistence:
    """One exception-safe journal snapshot, digest, and first write failure."""

    snapshot: ProviderAttemptJournalSnapshot
    artifact_sha256: str | None
    error: BaseException | None = None


@dataclass(frozen=True)
class Stage1ResponseEvaluation:
    """Typed accepted normalization or fail-closed Stage 1 rejection."""

    accepted: bool
    normalized: dict[str, Any] | None
    reason: Literal[
        "accepted",
        "missing_stage1_source_contract",
        "invalid_stage1_contract",
    ]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WholeAssetStructureEvaluation:
    """Typed whole-asset gate evaluated before per-component fan-out."""

    accepted: bool
    reason_codes: tuple[WholeAssetStructureReason, ...]
    robot_type: str | None
    dof: int | None
    segment_names: tuple[str, ...]
    source_prim_inventory: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        payload["segment_names"] = list(self.segment_names)
        payload["source_prim_inventory"] = list(self.source_prim_inventory)
        return payload


class ProviderAttemptJournal:
    """Single ordered, digest-bound persistence seam for provider attempts."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        on_persisted: Callable[[Path, str], None] | None = None,
        attempts: Iterable[Mapping[str, Any]] = (),
        whole_asset_structure: WholeAssetStructureEvaluation | None = None,
    ) -> None:
        self._path = path
        self._on_persisted = on_persisted
        self._attempts = [copy.deepcopy(dict(attempt)) for attempt in attempts]
        self._whole_asset_structure = whole_asset_structure
        self._failure_receipt_namespace = object()
        self._fallback_failure_receipts: weakref.WeakKeyDictionary[
            BaseException, ProviderAttemptReceipt
        ] = weakref.WeakKeyDictionary()
        self._lock = Lock()
        self._notification_lock = Lock()
        self._notification_state = local()
        self._latest_persisted_digest: str | None = None
        self._last_notified_digest: str | None = None

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        on_persisted: Callable[[Path, str], None] | None = None,
        expected_sha256: str | None = None,
    ) -> ProviderAttemptJournal:
        """Load a persisted journal, failing closed on malformed evidence."""
        raw_bytes = path.read_bytes()
        if expected_sha256 is not None:
            if not (
                isinstance(expected_sha256, str)
                and _SHA256_PATTERN.fullmatch(expected_sha256)
            ):
                raise ValueError("checkpointed provider-attempt digest is malformed")
            if hashlib.sha256(raw_bytes).hexdigest() != expected_sha256:
                raise _ProviderAttemptDigestMismatchError(
                    "provider-attempt journal checkpointed digest changed"
                )
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("provider-attempt journal must be an object")
        expected_top_level_fields = _PROVIDER_ATTEMPT_JOURNAL_FIELDS - (
            set()
            if payload.get("whole_asset_structure") is not None
            else {"whole_asset_structure"}
        )
        if set(payload) != expected_top_level_fields:
            raise ValueError("provider-attempt journal fields are malformed")
        raw_attempts = payload.get("attempts")
        if not isinstance(raw_attempts, list) or not all(
            isinstance(attempt, dict) for attempt in raw_attempts
        ):
            raise ValueError("attempts must be a list of objects")
        attempts: list[dict[str, Any]] = []
        for attempt in raw_attempts:
            request_kind = attempt.get("request_kind")
            outcome = attempt.get("outcome")
            attempt_number = attempt.get("attempt_number")
            raw_response = attempt.get("raw_response")
            error_type = attempt.get("error_type")
            normalized_diagnostics = attempt.get("normalized_diagnostics")
            entry_id = attempt.get("entry_id")
            expected_fields = _PROVIDER_ATTEMPT_FIELDS | (
                {"entry_id"} if entry_id is not None else set()
            )
            if not (
                set(attempt) == expected_fields
                and request_kind in _PROVIDER_REQUEST_KINDS
                and outcome in _PROVIDER_ATTEMPT_OUTCOMES
                and isinstance(attempt_number, int)
                and not isinstance(attempt_number, bool)
                and attempt_number > 0
                and (raw_response is None or isinstance(raw_response, dict))
                and (error_type is None or isinstance(error_type, str))
                and isinstance(normalized_diagnostics, dict)
                and (entry_id is None or isinstance(entry_id, str))
            ):
                raise ValueError("provider attempt evidence is malformed")
            if raw_response is not None:
                _validated_raw_provider_response_evidence(
                    raw_response,
                    label="provider raw-response evidence",
                )
            attempts.append(copy.deepcopy(attempt))
        structure_payload = payload.get("whole_asset_structure")
        structure = None
        if structure_payload is not None:
            if not isinstance(structure_payload, dict):
                raise ValueError("whole_asset_structure must be an object")
            canonical_structure_payload = dict(structure_payload)
            legacy_response_evidence = canonical_structure_payload.pop(
                "response_evidence", None
            )
            if legacy_response_evidence is not None:
                if not isinstance(legacy_response_evidence, dict):
                    raise ValueError("whole_asset_structure evidence is malformed")
                _validated_raw_provider_response_evidence(
                    legacy_response_evidence,
                    label="whole_asset_structure evidence",
                )
            structure = whole_asset_structure_evaluation_from_dict(
                canonical_structure_payload
            )
            if structure.to_dict() != canonical_structure_payload:
                raise ValueError("whole_asset_structure evidence is malformed")
        return cls(
            path=path,
            on_persisted=on_persisted,
            attempts=attempts,
            whole_asset_structure=structure,
        )

    def _snapshot_locked(self) -> ProviderAttemptJournalSnapshot:
        return ProviderAttemptJournalSnapshot(
            attempts=tuple(copy.deepcopy(attempt) for attempt in self._attempts),
            whole_asset_structure=self._whole_asset_structure,
        )

    def snapshot(self) -> ProviderAttemptJournalSnapshot:
        """Return an immutable copy safe for metadata and terminal projection."""
        with self._lock:
            return self._snapshot_locked()

    def attempt_count(self) -> int:
        """Return the ordered attempt count without copying retained evidence."""
        with self._lock:
            return len(self._attempts)

    def has_evidence(self) -> bool:
        """Return whether attempts or a current structure evaluation exist."""
        with self._lock:
            return bool(self._attempts or self._whole_asset_structure is not None)

    def attempt_for_failure(
        self,
        source_error: BaseException,
    ) -> Mapping[str, Any] | None:
        """Consume the exact attempt bound to an error or its wrapped cause."""
        error_chain = _explicit_error_chain(source_error)

        with self._lock:
            receipt: ProviderAttemptReceipt | None = None
            for error in error_chain:
                if (
                    isinstance(error, _ProviderAttemptReceiptCarrierError)
                    and error.namespace is self._failure_receipt_namespace
                    and not error.consumed
                ):
                    carrier_candidate = error.receipt
                    if self._receipt_is_current_locked(carrier_candidate):
                        receipt = carrier_candidate
                        error.consumed = True
                if receipt is not None:
                    break
                try:
                    receipts = getattr(error, _FAILURE_RECEIPTS_ATTRIBUTE, None)
                except Exception:
                    receipts = None
                attribute_candidate = (
                    receipts.get(self._failure_receipt_namespace)
                    if isinstance(receipts, dict)
                    else None
                )
                if isinstance(attribute_candidate, ProviderAttemptReceipt):
                    if isinstance(receipts, dict):
                        receipts.pop(self._failure_receipt_namespace, None)
                    if self._receipt_is_current_locked(attribute_candidate):
                        receipt = attribute_candidate
                if receipt is not None:
                    break
                try:
                    fallback_candidate = self._fallback_failure_receipts.get(error)
                    if fallback_candidate is not None:
                        del self._fallback_failure_receipts[error]
                except TypeError:
                    fallback_candidate = None
                if isinstance(
                    fallback_candidate, ProviderAttemptReceipt
                ) and self._receipt_is_current_locked(fallback_candidate):
                    receipt = fallback_candidate
                if receipt is not None:
                    break
            if receipt is None:
                return None
            return copy.deepcopy(self._attempts[receipt.sequence_number - 1])

    def _receipt_is_current_locked(self, receipt: ProviderAttemptReceipt) -> bool:
        """Validate one receipt against durable ordering without retained errors."""
        sequence_number = receipt.sequence_number
        if not 1 <= sequence_number <= len(self._attempts):
            return False
        attempt = self._attempts[sequence_number - 1]
        if not (
            attempt.get("outcome") == "transport_error"
            and attempt.get("entry_id") == receipt.entry_id
        ):
            return False
        return not any(
            later_attempt.get("entry_id") == receipt.entry_id
            and later_attempt.get("outcome") != "transport_error"
            for later_attempt in self._attempts[sequence_number:]
        )

    def latest_transport_attempt(
        self,
        entry_ids: Iterable[str],
        *,
        minimum_sequence_number: int = 1,
    ) -> Mapping[str, Any] | None:
        """Return an unresolved entry's latest exact transport outcome."""
        if (
            isinstance(minimum_sequence_number, bool)
            or not isinstance(minimum_sequence_number, int)
            or minimum_sequence_number < 1
        ):
            raise ValueError("minimum_sequence_number must be a positive integer")
        ordered_ids = tuple(dict.fromkeys(entry_ids))
        wanted = set(ordered_ids)
        with self._lock:
            latest_by_entry: dict[str, Mapping[str, Any]] = {}
            for sequence_number, attempt in enumerate(self._attempts, start=1):
                if sequence_number < minimum_sequence_number:
                    continue
                entry_id = attempt.get("entry_id")
                if isinstance(entry_id, str) and entry_id in wanted:
                    latest_by_entry[entry_id] = attempt
            for entry_id in ordered_ids:
                latest = latest_by_entry.get(entry_id)
                if latest is not None and latest.get("outcome") == "transport_error":
                    return copy.deepcopy(latest)
        return None

    def _persist_locked(self) -> str | None:
        if self._path is None:
            return None
        payload = (
            json.dumps(
                self._snapshot_locked().to_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_bytes(payload)
            temporary_path.replace(self._path)
        finally:
            temporary_path.unlink(missing_ok=True)
        digest = hashlib.sha256(payload).hexdigest()
        self._latest_persisted_digest = digest
        return digest

    def _notify_persisted(self, digest: str | None) -> None:
        if digest is None or self._on_persisted is None or self._path is None:
            return
        if getattr(self._notification_state, "active", False):
            return
        self._notification_lock.acquire()
        try:
            self._notification_state.active = True
            while True:
                with self._lock:
                    pending_digest = self._latest_persisted_digest
                if (
                    pending_digest is None
                    or pending_digest == self._last_notified_digest
                ):
                    break
                self._on_persisted(self._path, pending_digest)
                self._last_notified_digest = pending_digest
        finally:
            self._notification_state.active = False
            self._notification_lock.release()

    def persist(self) -> str | None:
        """Persist the current snapshot and return its SHA-256 digest."""
        with self._lock:
            digest = self._persist_locked()
        self._notify_persisted(digest)
        return digest

    def persisted_snapshot(
        self,
    ) -> tuple[ProviderAttemptJournalSnapshot, str | None]:
        """Atomically persist and return the exact snapshot bound to its digest."""
        with self._lock:
            snapshot = self._snapshot_locked()
            digest = self._persist_locked()
        self._notify_persisted(digest)
        return snapshot, digest

    def _discard_resolved_failures_locked(
        self,
        payloads: Iterable[Mapping[str, Any]],
    ) -> None:
        resolved_entry_ids = {
            payload.get("entry_id")
            for payload in payloads
            if payload.get("outcome") != "transport_error"
        }
        if not resolved_entry_ids:
            return
        for error, receipt in tuple(self._fallback_failure_receipts.items()):
            if receipt.entry_id in resolved_entry_ids:
                self._fallback_failure_receipts.pop(error, None)

    def _bind_failure_receipt_locked(
        self,
        source_error: BaseException,
        receipt: ProviderAttemptReceipt,
    ) -> _ProviderAttemptReceiptCarrierError | None:
        """Bind without retaining an active provider exception in the journal."""
        try:
            raw_receipts = getattr(
                source_error,
                _FAILURE_RECEIPTS_ATTRIBUTE,
                None,
            )
            receipts = dict(raw_receipts) if isinstance(raw_receipts, dict) else {}
            receipts[self._failure_receipt_namespace] = receipt
            setattr(source_error, _FAILURE_RECEIPTS_ATTRIBUTE, receipts)
        except Exception:
            try:
                self._fallback_failure_receipts[source_error] = receipt
            except TypeError:
                carrier = _ProviderAttemptReceiptCarrierError(
                    self._failure_receipt_namespace,
                    receipt,
                )
                return carrier
        else:
            try:
                self._fallback_failure_receipts.pop(
                    source_error,
                    None,
                )
            except TypeError:
                pass
        return None

    def record(
        self,
        diagnostic: ProviderAttemptDiagnostic,
        *,
        entry_id: str | None = None,
        source_error: BaseException | None = None,
        _preserve_source_error: bool = False,
    ) -> ProviderAttemptRecording:
        """Append one typed attempt and return its exact in-process identity."""
        payload = diagnostic.to_dict()
        if entry_id is not None:
            payload = {"entry_id": entry_id, **payload}
        persistence_error: BaseException | None = None
        receipt_carrier: _ProviderAttemptReceiptCarrierError | None = None
        with self._lock:
            self._attempts.append(payload)
            self._discard_resolved_failures_locked((payload,))
            receipt = ProviderAttemptReceipt(
                sequence_number=len(self._attempts),
                entry_id=entry_id,
            )
            if source_error is not None:
                receipt_carrier = self._bind_failure_receipt_locked(
                    source_error,
                    receipt,
                )
            try:
                digest = self._persist_locked()
            except Exception as error:
                if not (_preserve_source_error and source_error is not None):
                    raise ProviderAttemptRecorderError(
                        f"provider attempt persistence failed: {type(error).__name__}",
                        persistence_error=error,
                    ) from error
                digest = None
                persistence_error = error
        try:
            self._notify_persisted(digest)
        except Exception as error:
            if not (_preserve_source_error and source_error is not None):
                raise ProviderAttemptRecorderError(
                    f"provider attempt persistence failed: {type(error).__name__}",
                    persistence_error=error,
                ) from error
            persistence_error = error
        if persistence_error is not None and source_error is not None:
            _try_add_exception_note(
                source_error,
                "Provider-attempt persistence failed while retaining the "
                f"provider error: {type(persistence_error).__name__}",
            )
            recorder_error = ProviderAttemptRecorderError(
                "provider attempt persistence failed: "
                f"{type(persistence_error).__name__}",
                persistence_error=persistence_error,
            )
            if receipt_carrier is not None:
                receipt_carrier.__cause__ = source_error
                raise recorder_error from receipt_carrier
            raise recorder_error from source_error
        return ProviderAttemptRecording(
            receipt=receipt,
            failure_carrier=receipt_carrier,
        )

    def record_transport_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        entry_id: str | None = None,
        initial_request_kind: ProviderRequestKind = "initial",
    ) -> ProviderAttemptRecording:
        """Project one provider callback into the canonical attempt contract."""
        raw_attempt_number = attempt.get("attempt_number", 1)
        if (
            not isinstance(raw_attempt_number, int)
            or isinstance(raw_attempt_number, bool)
            or raw_attempt_number < 1
        ):
            raise ValueError("provider attempt_number must be a positive integer")
        attempt_number = raw_attempt_number
        raw_request_kind = attempt.get("request_kind")
        request_kind = cast(
            ProviderRequestKind,
            raw_request_kind
            if raw_request_kind is not None
            else ("transport_retry" if attempt_number > 1 else initial_request_kind),
        )
        if request_kind not in _PROVIDER_REQUEST_KINDS:
            raise ValueError("provider request_kind is unsupported")
        outcome = attempt.get("outcome", "transport_error")
        if outcome not in _PROVIDER_ATTEMPT_OUTCOMES:
            raise ValueError("provider attempt outcome is unsupported")
        source_error = attempt.get("error")
        return self.record(
            build_provider_attempt_diagnostic(
                request_kind=request_kind,
                outcome=cast(ProviderAttemptOutcome, outcome),
                attempt_number=attempt_number,
                raw_response=(
                    cast(str, attempt["raw_response"])
                    if isinstance(attempt.get("raw_response"), str)
                    else None
                ),
                error=(
                    source_error if isinstance(source_error, BaseException) else None
                ),
            ),
            entry_id=entry_id,
            source_error=(
                source_error if isinstance(source_error, BaseException) else None
            ),
            _preserve_source_error=True,
        )

    def record_many(
        self,
        diagnostics: Iterable[tuple[str | None, ProviderAttemptDiagnostic]],
    ) -> None:
        """Append a deterministic group of typed attempts and persist once."""
        payloads: list[dict[str, Any]] = []
        for entry_id, diagnostic in diagnostics:
            payload = diagnostic.to_dict()
            if entry_id is not None:
                payload = {"entry_id": entry_id, **payload}
            payloads.append(payload)
        if not payloads:
            return
        try:
            with self._lock:
                self._attempts.extend(payloads)
                self._discard_resolved_failures_locked(payloads)
                digest = self._persist_locked()
            self._notify_persisted(digest)
        except Exception as error:
            if isinstance(error, ProviderAttemptRecorderError):
                raise
            raise ProviderAttemptRecorderError(
                f"provider attempt persistence failed: {type(error).__name__}",
                persistence_error=error,
            ) from error

    def set_whole_asset_structure(
        self,
        evaluation: WholeAssetStructureEvaluation,
    ) -> None:
        """Project the whole-asset gate once and persist it with its attempts."""
        try:
            with self._lock:
                self._whole_asset_structure = evaluation
                digest = self._persist_locked()
            self._notify_persisted(digest)
        except Exception as error:
            if isinstance(error, ProviderAttemptRecorderError):
                raise
            raise ProviderAttemptRecorderError(
                f"provider attempt persistence failed: {type(error).__name__}",
                persistence_error=error,
            ) from error

    def clear_whole_asset_structure(self) -> None:
        """Clear a prior invocation's verdict while preserving its attempts."""
        try:
            with self._lock:
                self._whole_asset_structure = None
                digest = self._persist_locked()
            self._notify_persisted(digest)
        except Exception as error:
            if isinstance(error, ProviderAttemptRecorderError):
                raise
            raise ProviderAttemptRecorderError(
                f"provider attempt persistence failed: {type(error).__name__}",
                persistence_error=error,
            ) from error


def persist_provider_attempt_journal(
    journal: ProviderAttemptJournal,
    *,
    prior_error: BaseException | None = None,
) -> ProviderAttemptPersistence:
    """Persist once while retaining the first write/callback failure for status."""
    projected_prior_error = (
        provider_attempt_persistence_error(prior_error) or prior_error
        if prior_error is not None
        else None
    )
    try:
        snapshot, digest = journal.persisted_snapshot()
    except Exception as current_error:
        return ProviderAttemptPersistence(
            snapshot=journal.snapshot(),
            artifact_sha256=None,
            error=(
                projected_prior_error
                if projected_prior_error is not None
                else provider_attempt_persistence_error(current_error) or current_error
            ),
        )
    return ProviderAttemptPersistence(
        snapshot=snapshot,
        artifact_sha256=digest,
        error=projected_prior_error,
    )


class ProviderResponseConformanceTerminalError(RuntimeError):
    """Typed terminal failure consumable by fixed-pipeline and embedded adapters."""

    @staticmethod
    def _build_status(
        *,
        error_type: str,
        failure_stage: str,
        reason: str,
        attempt_diagnostics: Iterable[Mapping[str, Any]],
        diagnostics_artifact_path: str | None,
        diagnostics_artifact_sha256: str | None,
        diagnostics_persistence_error: BaseException | None,
    ) -> dict[str, Any]:
        status: dict[str, Any] = {
            "requested": True,
            "attempted": True,
            "accepted": False,
            "outcome": "failed",
            "failure_stage": failure_stage,
            "error_type": error_type,
            "reason": reason,
            "attempt_diagnostics": [dict(item) for item in attempt_diagnostics],
            "diagnostics_artifact_status": (
                "persisted"
                if diagnostics_artifact_sha256 is not None
                else "unavailable"
            ),
        }
        if diagnostics_artifact_sha256 is not None:
            if diagnostics_artifact_path is not None:
                status["diagnostics_artifact_path"] = diagnostics_artifact_path
            status["diagnostics_artifact_sha256"] = diagnostics_artifact_sha256
        if diagnostics_persistence_error is not None:
            status["diagnostics_persistence_error_type"] = type(
                diagnostics_persistence_error
            ).__name__
        return status

    @classmethod
    def _from_parts(
        cls,
        *,
        failure_stage: str,
        reason: str,
        attempt_diagnostics: Iterable[Mapping[str, Any]],
        diagnostics_artifact_path: str | None,
        diagnostics_artifact_sha256: str | None,
        diagnostics_persistence_error: BaseException | None,
        message: str,
    ) -> ProviderResponseConformanceTerminalError:
        error = cls.__new__(cls)
        error.status = cls._build_status(
            error_type=cls.__name__,
            failure_stage=failure_stage,
            reason=reason,
            attempt_diagnostics=attempt_diagnostics,
            diagnostics_artifact_path=diagnostics_artifact_path,
            diagnostics_artifact_sha256=diagnostics_artifact_sha256,
            diagnostics_persistence_error=diagnostics_persistence_error,
        )
        RuntimeError.__init__(error, message)
        return error

    def __init__(
        self,
        evaluation: WholeAssetStructureEvaluation,
        *,
        diagnostics_artifact_path: str | None = None,
        diagnostics_artifact_sha256: str | None = None,
        attempt_diagnostics: list[dict[str, Any]] | None = None,
        diagnostics_persistence_error: BaseException | None = None,
    ) -> None:
        if evaluation.accepted or not evaluation.reason_codes:
            raise ValueError("terminal conformance error requires a rejection")
        self.status = self._build_status(
            error_type=type(self).__name__,
            failure_stage="whole_asset_structure",
            reason=evaluation.reason_codes[0],
            attempt_diagnostics=(
                list(attempt_diagnostics)
                if attempt_diagnostics is not None
                else [evaluation.to_dict()]
            ),
            diagnostics_artifact_path=diagnostics_artifact_path,
            diagnostics_artifact_sha256=diagnostics_artifact_sha256,
            diagnostics_persistence_error=diagnostics_persistence_error,
        )
        super().__init__(
            "Provider-backed whole-asset structure failed closed: "
            + ", ".join(evaluation.reason_codes)
        )

    @classmethod
    def for_stage1(
        cls,
        *,
        reason: str,
        attempt_diagnostics: list[dict[str, Any]],
        diagnostics_artifact_path: str | None,
        diagnostics_artifact_sha256: str | None,
        diagnostics_persistence_error: BaseException | None = None,
        message: str | None = None,
    ) -> ProviderResponseConformanceTerminalError:
        """Build the same typed terminal boundary for Stage 1 rejection."""
        return cls._from_parts(
            failure_stage="stage1_response_contract",
            reason=reason,
            attempt_diagnostics=attempt_diagnostics,
            diagnostics_artifact_path=diagnostics_artifact_path,
            diagnostics_artifact_sha256=diagnostics_artifact_sha256,
            diagnostics_persistence_error=diagnostics_persistence_error,
            message=(
                message or f"Provider-backed Stage 1 response failed closed: {reason}"
            ),
        )

    @classmethod
    def for_transport(
        cls,
        *,
        failure_stage: str,
        attempt_diagnostics: list[dict[str, Any]],
        diagnostics_artifact_path: str | None,
        diagnostics_artifact_sha256: str | None,
        source_error: BaseException | None = None,
        source_error_type: str | None = None,
        diagnostics_persistence_error: BaseException | None = None,
    ) -> ProviderResponseConformanceTerminalError:
        """Build the shared terminal boundary for exhausted transport retries."""
        provider_error_type = source_error_type
        if provider_error_type is None and source_error is not None:
            provider_error_type = type(source_error).__name__
        return cls._from_parts(
            failure_stage=failure_stage,
            reason="provider_transport_exhausted",
            attempt_diagnostics=attempt_diagnostics,
            diagnostics_artifact_path=diagnostics_artifact_path,
            diagnostics_artifact_sha256=diagnostics_artifact_sha256,
            diagnostics_persistence_error=diagnostics_persistence_error,
            message=(
                "Provider-backed request exhausted transport retries: "
                f"{provider_error_type or 'ProviderTransportError'}"
            ),
        )

    @classmethod
    def for_evidence(
        cls,
        *,
        failure_stage: str,
        reason: str,
        attempt_diagnostics: list[dict[str, Any]],
        diagnostics_artifact_path: str | None = None,
        diagnostics_artifact_sha256: str | None = None,
        diagnostics_persistence_error: BaseException | None = None,
        message: str | None = None,
    ) -> ProviderResponseConformanceTerminalError:
        """Build the typed terminal for unavailable or invalid provider evidence."""
        return cls._from_parts(
            failure_stage=failure_stage,
            reason=reason,
            attempt_diagnostics=attempt_diagnostics,
            diagnostics_artifact_path=diagnostics_artifact_path,
            diagnostics_artifact_sha256=diagnostics_artifact_sha256,
            diagnostics_persistence_error=diagnostics_persistence_error,
            message=message or f"Provider-backed evidence failed closed: {reason}",
        )

    @classmethod
    def from_status(
        cls,
        status: Mapping[str, Any],
    ) -> ProviderResponseConformanceTerminalError:
        """Rehydrate the typed failure across a workflow exception boundary."""
        error = cls.__new__(cls)
        error.status = dict(status)
        error.status["attempt_diagnostics"] = list(
            status.get("attempt_diagnostics") or ()
        )
        failure_stage = str(status.get("failure_stage") or "provider_response")
        RuntimeError.__init__(
            error,
            f"Provider-backed conformance failed closed at {failure_stage}",
        )
        return error


def _raise_terminal_without_provider_chain(
    error: ProviderResponseConformanceTerminalError,
) -> NoReturn:
    """Raise a typed terminal without retaining provider exception state."""
    error.__cause__ = None
    error.__context__ = None
    error.__traceback__ = None
    try:
        # CPython repopulates ``__context__`` when this helper is called while
        # handling a provider error. Catch the typed error locally, detach the
        # newly populated context, and use a bare re-raise so the object that
        # crosses the workflow boundary owns no provider exception or frames.
        raise error from None
    except ProviderResponseConformanceTerminalError as raised:
        raised.__cause__ = None
        raised.__context__ = None
        raised.__traceback__ = None
        raise


def _cleared_provider_call() -> None:
    """Replace provider-bearing call closures before a typed terminal escapes."""


def require_provider_attempt_journal_persistence(
    journal: ProviderAttemptJournal,
    *,
    failure_stage: str,
    diagnostics_artifact_path: str,
    on_terminal: Callable[[ProviderResponseConformanceTerminalError], None]
    | None = None,
    attempt_diagnostics: Iterable[Mapping[str, Any]] = (),
    prior_error: BaseException | None = None,
) -> ProviderAttemptPersistence:
    """Require durable evidence or raise the canonical typed evidence terminal."""
    persistence = persist_provider_attempt_journal(
        journal,
        prior_error=prior_error,
    )
    if persistence.error is None and persistence.artifact_sha256 is not None:
        return persistence
    persistence_error = persistence.error or RuntimeError(
        "provider-attempt journal has no durable artifact path"
    )
    terminal_error = ProviderResponseConformanceTerminalError.for_evidence(
        failure_stage=failure_stage,
        reason="provider_evidence_persistence_failed",
        attempt_diagnostics=[dict(diagnostic) for diagnostic in attempt_diagnostics],
        diagnostics_artifact_path=(
            diagnostics_artifact_path
            if persistence.artifact_sha256 is not None
            else None
        ),
        diagnostics_artifact_sha256=persistence.artifact_sha256,
        diagnostics_persistence_error=persistence_error,
    )
    if on_terminal is not None:
        on_terminal(terminal_error)
    _raise_terminal_without_provider_chain(terminal_error)


def load_provider_attempt_journal(
    path: Path,
    *,
    on_persisted: Callable[[Path, str], None] | None,
    expected_sha256: str | None,
    failure_stage: str,
    on_terminal: Callable[[ProviderResponseConformanceTerminalError], None]
    | None = None,
) -> ProviderAttemptJournal:
    """Load resume evidence or emit one typed, non-resumable evidence terminal."""
    try:
        return ProviderAttemptJournal.load(
            path,
            on_persisted=on_persisted,
            expected_sha256=expected_sha256,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as source_error:
        if isinstance(source_error, FileNotFoundError):
            reason = "provider_evidence_missing"
        elif isinstance(source_error, _ProviderAttemptDigestMismatchError):
            reason = "provider_evidence_digest_mismatch"
        else:
            reason = "provider_evidence_invalid"
        terminal_error = ProviderResponseConformanceTerminalError.for_evidence(
            failure_stage=failure_stage,
            reason=reason,
            attempt_diagnostics=[
                {
                    "reason": reason,
                    "error_type": type(source_error).__name__,
                }
            ],
            message=(
                f"Cannot resume from trusted provider-response evidence: {reason}"
            ),
        )
        if on_terminal is not None:
            on_terminal(terminal_error)
        _raise_terminal_without_provider_chain(terminal_error)


def evaluate_exhausted_transport_terminal(
    *,
    journal: ProviderAttemptJournal,
    failure_stage: str,
    diagnostics_artifact_path: str,
    source_error: BaseException | None = None,
    entry_ids: Iterable[str] = (),
    minimum_sequence_number: int = 1,
    on_terminal: Callable[[ProviderResponseConformanceTerminalError], None]
    | None = None,
) -> ProviderResponseConformanceTerminalError | None:
    """Project exception or returned-row transport exhaustion through one seam."""
    if source_error is not None:
        failure_attempt = journal.attempt_for_failure(source_error)
    else:
        failure_attempt = journal.latest_transport_attempt(
            entry_ids,
            minimum_sequence_number=minimum_sequence_number,
        )
    if failure_attempt is None or failure_attempt.get("outcome") != "transport_error":
        return None
    persistence = persist_provider_attempt_journal(journal)
    diagnostics_sha256 = persistence.artifact_sha256
    persistence_error = persistence.error
    if persistence_error is None and source_error is not None:
        persistence_error = provider_attempt_persistence_error(source_error)
    provider_error_type = failure_attempt.get("error_type")
    terminal_error = ProviderResponseConformanceTerminalError.for_transport(
        failure_stage=failure_stage,
        attempt_diagnostics=[dict(failure_attempt)],
        diagnostics_artifact_path=(
            diagnostics_artifact_path if diagnostics_sha256 is not None else None
        ),
        diagnostics_artifact_sha256=diagnostics_sha256,
        source_error_type=(
            provider_error_type
            if isinstance(provider_error_type, str) and provider_error_type
            else type(source_error).__name__
            if source_error is not None
            else "ProviderTransportError"
        ),
        diagnostics_persistence_error=persistence_error,
    )
    if on_terminal is not None:
        on_terminal(terminal_error)
    return terminal_error


def run_provider_call_with_journal(
    call: Callable[[], Any],
    *,
    journal: ProviderAttemptJournal,
    failure_stage: str,
    diagnostics_artifact_path: str,
    on_terminal: Callable[[ProviderResponseConformanceTerminalError], None]
    | None = None,
    capture_attempt: bool = False,
    entry_id: str | None = None,
    request_kind: ProviderRequestKind = "initial",
) -> Any:
    """Run one provider surface with shared persistence and terminal semantics."""
    try:
        result = call()
    except ProviderResponseConformanceTerminalError:
        raise
    except Exception as source_error:
        terminal_source_error: BaseException = source_error
        if capture_attempt:
            try:
                recording = journal.record_transport_attempt(
                    {
                        "request_kind": request_kind,
                        "outcome": "transport_error",
                        "attempt_number": 1,
                        "error": source_error,
                    },
                    entry_id=entry_id,
                )
                if recording.failure_carrier is not None:
                    terminal_source_error = recording.failure_carrier
            except (
                ProviderAttemptRecorderError,
                _ProviderAttemptReceiptCarrierError,
            ) as recorder_error:
                # The attempt remains in memory even when its durable write
                # fails or needs an external identity carrier. Route that exact
                # attempt through the same typed terminal evaluator as callbacks.
                terminal_source_error = recorder_error
        terminal_error = evaluate_exhausted_transport_terminal(
            journal=journal,
            failure_stage=failure_stage,
            diagnostics_artifact_path=diagnostics_artifact_path,
            source_error=terminal_source_error,
            on_terminal=on_terminal,
        )
        if terminal_error is not None:
            # The typed terminal traceback retains this caller frame. Remove
            # provider-bearing locals before raising so traceback-local
            # capture cannot recover response bodies or credentials after the
            # public exception chain has been detached.
            del terminal_source_error
            del source_error
            call = _cleared_provider_call
            _raise_terminal_without_provider_chain(terminal_error)
        recorder_persistence_error = provider_attempt_persistence_error(
            terminal_source_error
        )
        if recorder_persistence_error is not None:
            snapshot = journal.snapshot()
            try:
                require_provider_attempt_journal_persistence(
                    journal,
                    failure_stage=failure_stage,
                    diagnostics_artifact_path=diagnostics_artifact_path,
                    on_terminal=on_terminal,
                    attempt_diagnostics=snapshot.attempts[-1:],
                    prior_error=recorder_persistence_error,
                )
            except ProviderResponseConformanceTerminalError as persistence_terminal:
                del terminal_source_error
                del source_error
                call = _cleared_provider_call
                _raise_terminal_without_provider_chain(persistence_terminal)
        try:
            journal.persist()
        except Exception as persistence_error:
            snapshot = journal.snapshot()
            try:
                require_provider_attempt_journal_persistence(
                    journal,
                    failure_stage=failure_stage,
                    diagnostics_artifact_path=diagnostics_artifact_path,
                    on_terminal=on_terminal,
                    attempt_diagnostics=snapshot.attempts[-1:],
                    prior_error=(
                        provider_attempt_persistence_error(persistence_error)
                        or persistence_error
                    ),
                )
            except ProviderResponseConformanceTerminalError as persistence_terminal:
                del terminal_source_error
                del source_error
                call = _cleared_provider_call
                _raise_terminal_without_provider_chain(persistence_terminal)
        raise
    if capture_attempt:
        response_is_text = isinstance(result, str)
        if response_is_text:
            raw_response = result
            outcome: ProviderAttemptOutcome = (
                "response_received" if raw_response.strip() else "empty_response"
            )
            normalized_diagnostics = None
        else:
            raw_response = None
            outcome = "contract_rejected"
            normalized_diagnostics = {
                "reason": "invalid_provider_response_type",
                "response_type": type(result).__name__,
            }
        diagnostic = build_provider_attempt_diagnostic(
            request_kind=request_kind,
            outcome=outcome,
            attempt_number=1,
            raw_response=raw_response,
            normalized_diagnostics=normalized_diagnostics,
        )
        try:
            journal.record(
                diagnostic,
                entry_id=entry_id,
            )
        except Exception as persistence_error:
            snapshot = journal.snapshot()
            try:
                require_provider_attempt_journal_persistence(
                    journal,
                    failure_stage=failure_stage,
                    diagnostics_artifact_path=diagnostics_artifact_path,
                    on_terminal=on_terminal,
                    attempt_diagnostics=snapshot.attempts[-1:],
                    prior_error=(
                        provider_attempt_persistence_error(persistence_error)
                        or persistence_error
                    ),
                )
            except ProviderResponseConformanceTerminalError as persistence_terminal:
                result = None
                raw_response = None
                call = _cleared_provider_call
                _raise_terminal_without_provider_chain(persistence_terminal)
        if not response_is_text:
            persistence = persist_provider_attempt_journal(journal)
            terminal_error = ProviderResponseConformanceTerminalError.for_evidence(
                failure_stage=failure_stage,
                reason="invalid_provider_response_type",
                attempt_diagnostics=[dict(persistence.snapshot.attempts[-1])],
                diagnostics_artifact_path=(
                    diagnostics_artifact_path
                    if persistence.artifact_sha256 is not None
                    else None
                ),
                diagnostics_artifact_sha256=persistence.artifact_sha256,
                diagnostics_persistence_error=persistence.error,
                message="Provider-backed request returned a non-text response",
            )
            if on_terminal is not None:
                on_terminal(terminal_error)
            result = None
            call = _cleared_provider_call
            _raise_terminal_without_provider_chain(terminal_error)
    return result


def _bounded_bytes(value: bytes, *, max_bytes: int) -> bytes:
    if len(value) <= max_bytes:
        return value
    marker = b"\n...[bounded response truncated]...\n"
    available = max_bytes - len(marker)
    head = available // 2
    tail = available - head
    return value[:head] + marker + value[-tail:]


def raw_provider_response_evidence(
    response_text: str,
    *,
    max_retained_bytes: int | None = None,
) -> RawProviderResponseEvidence:
    """Return exact full-response identity plus bounded safe replay evidence."""
    if max_retained_bytes is None:
        max_retained_bytes = _MAX_RETAINED_RESPONSE_BYTES
    marker_bytes = len(b"\n...[bounded response truncated]...\n")
    if (
        isinstance(max_retained_bytes, bool)
        or not isinstance(max_retained_bytes, int)
        or max_retained_bytes <= marker_bytes
    ):
        raise ValueError("max_retained_bytes cannot fit the truncation marker")
    if max_retained_bytes > _MAX_RETAINED_RESPONSE_BYTES:
        raise ValueError("max_retained_bytes cannot exceed the journal evidence limit")
    response_bytes = response_text.encode("utf-8", errors="replace")
    projected = redact_sensitive_config(response_text)
    safe_text = projected if isinstance(projected, str) else "<redacted>"
    safe_bytes = safe_text.encode("utf-8", errors="replace")
    retained = _bounded_bytes(safe_bytes, max_bytes=max_retained_bytes)
    retained_text = retained.decode("utf-8", errors="ignore")
    retained = retained_text.encode("utf-8")
    return RawProviderResponseEvidence(
        response_sha256=hashlib.sha256(response_bytes).hexdigest(),
        response_bytes=len(response_bytes),
        retained_response=retained_text,
        retained_response_sha256=hashlib.sha256(retained).hexdigest(),
        retained_bytes=len(retained),
        truncated=len(safe_bytes) > len(retained),
        redacted=safe_text != response_text,
    )


def build_provider_attempt_diagnostic(
    *,
    request_kind: ProviderRequestKind,
    outcome: ProviderAttemptOutcome,
    attempt_number: int,
    raw_response: str | None = None,
    error: BaseException | None = None,
    normalized_diagnostics: Mapping[str, Any] | None = None,
) -> ProviderAttemptDiagnostic:
    """Build a bounded value-safe attempt diagnostic."""
    if attempt_number < 1:
        raise ValueError("attempt_number must be positive")
    return ProviderAttemptDiagnostic(
        request_kind=request_kind,
        outcome=outcome,
        attempt_number=attempt_number,
        raw_response=(
            raw_provider_response_evidence(raw_response)
            if raw_response is not None
            else None
        ),
        error_type=type(error).__name__ if error is not None else None,
        normalized_diagnostics=dict(normalized_diagnostics or {}),
    )


def evaluate_stage1_response(
    payload: Any,
    *,
    output_key: str = "classification",
) -> Stage1ResponseEvaluation:
    """Normalize only documented Stage 1 shapes and retain typed diagnostics."""
    if not isinstance(payload, dict) or not has_parseable_stage1_source_response(
        payload,
        output_key=output_key,
    ):
        return Stage1ResponseEvaluation(
            accepted=False,
            normalized=None,
            reason="missing_stage1_source_contract",
            diagnostics={"input_type": type(payload).__name__},
        )
    try:
        normalized = normalize_stage1_prediction_payload(
            payload,
            output_key=output_key,
        )
    except (TypeError, ValueError, ValidationError) as error:
        return Stage1ResponseEvaluation(
            accepted=False,
            normalized=None,
            reason="invalid_stage1_contract",
            diagnostics={"error_type": type(error).__name__},
        )
    if not isinstance(normalized, dict):
        return Stage1ResponseEvaluation(
            accepted=False,
            normalized=None,
            reason="invalid_stage1_contract",
            diagnostics={"error_type": "NormalizedPayloadTypeError"},
        )
    return Stage1ResponseEvaluation(
        accepted=True,
        normalized=normalized,
        reason="accepted",
    )


def _extract_structure_object(response_text: str) -> dict[str, Any] | None:
    answer_match = re.search(
        r"<answer[^>]*>(.*?)</answer>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    candidate = answer_match.group(1).strip() if answer_match else response_text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        object_match = re.search(r"\{[\s\S]*\}", candidate)
        if object_match is None:
            return None
        try:
            parsed = json.loads(object_match.group())
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _taxonomy_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", value.lower()):
        if token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if token not in _STRUCTURE_GENERIC_TOKENS:
            tokens.add(token)
    return tokens


def _taxonomy_is_compatible(
    robot_type: str,
    *,
    asset_type: str | None,
    asset_subtype: str | None,
) -> bool:
    normalized_robot_type = robot_type.strip().lower()
    if normalized_robot_type in _UNACCEPTABLE_TAXONOMY:
        return False
    identified_text = " ".join(
        value for value in (asset_type, asset_subtype) if isinstance(value, str)
    )
    expected = _taxonomy_tokens(identified_text)
    observed = _taxonomy_tokens(robot_type)
    return not expected or bool(expected & observed)


def evaluate_whole_asset_structure(
    response_text: str,
    *,
    asset_type: str | None,
    asset_subtype: str | None,
    articulation_intended: bool,
    source_prim_inventory: tuple[str, ...] = (),
) -> WholeAssetStructureEvaluation:
    """Evaluate one whole-asset response before any component fan-out."""
    parsed = _extract_structure_object(response_text)
    if not articulation_intended:
        optional = parsed or {}
        optional_robot_type = optional.get("robot_type")
        optional_dof = optional.get("dof")
        optional_names = optional.get("segment_names")
        valid_optional_names = bool(
            isinstance(optional_names, list)
            and all(isinstance(name, str) and name.strip() for name in optional_names)
        )
        return WholeAssetStructureEvaluation(
            accepted=True,
            reason_codes=(),
            robot_type=(
                optional_robot_type
                if isinstance(optional_robot_type, str) and optional_robot_type.strip()
                else None
            ),
            dof=(
                optional_dof
                if isinstance(optional_dof, int)
                and not isinstance(optional_dof, bool)
                and optional_dof >= 0
                else None
            ),
            segment_names=(
                tuple(cast(list[str], optional_names)) if valid_optional_names else ()
            ),
            source_prim_inventory=tuple(source_prim_inventory),
        )

    if parsed is None:
        return WholeAssetStructureEvaluation(
            accepted=False,
            reason_codes=("missing_whole_asset_source_contract",),
            robot_type=None,
            dof=None,
            segment_names=(),
            source_prim_inventory=tuple(source_prim_inventory),
        )

    robot_type = parsed.get("robot_type")
    dof = parsed.get("dof")
    names = parsed.get("segment_names")
    valid_shape = bool(
        isinstance(robot_type, str)
        and robot_type.strip()
        and isinstance(dof, int)
        and not isinstance(dof, bool)
        and dof >= 0
        and isinstance(names, list)
        and all(isinstance(name, str) and name.strip() for name in names)
    )
    if not valid_shape:
        return WholeAssetStructureEvaluation(
            accepted=False,
            reason_codes=("missing_whole_asset_source_contract",),
            robot_type=robot_type if isinstance(robot_type, str) else None,
            dof=dof if isinstance(dof, int) and not isinstance(dof, bool) else None,
            segment_names=(),
            source_prim_inventory=tuple(source_prim_inventory),
        )

    robot_type = cast(str, robot_type)
    dof = cast(int, dof)
    segment_names = tuple(cast(list[str], names))
    reasons: list[Any] = []
    # A coherent zero-DOF result is a valid, evidence-backed finding for a
    # legitimately non-articulated asset. Mixed zero/nonzero structure remains
    # invalid because it cannot safely drive component fan-out.
    taxonomy_compatible = _taxonomy_is_compatible(
        robot_type,
        asset_type=asset_type,
        asset_subtype=asset_subtype,
    )
    coherent_non_articulated = (
        dof == 0 and not segment_names and bool(source_prim_inventory)
    )
    if (dof == 0 or not segment_names) and not (
        coherent_non_articulated and taxonomy_compatible
    ):
        reasons.append("zero_whole_asset_assignments")
    if not taxonomy_compatible:
        reasons.append("incompatible_whole_asset_taxonomy")
    return WholeAssetStructureEvaluation(
        accepted=not reasons,
        reason_codes=tuple(reasons),
        robot_type=robot_type,
        dof=dof,
        segment_names=segment_names,
        source_prim_inventory=tuple(source_prim_inventory),
    )


def require_whole_asset_structure(
    evaluation: WholeAssetStructureEvaluation | None,
    *,
    diagnostics_artifact_path: str | None = None,
    diagnostics_artifact_sha256: str | None = None,
    attempt_diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_persistence_error: BaseException | None = None,
) -> None:
    """Raise the shared typed terminal error for a required rejection."""
    if evaluation is None:
        raise ProviderResponseConformanceTerminalError.for_evidence(
            failure_stage="whole_asset_structure_evidence",
            reason="provider_evidence_missing",
            attempt_diagnostics=attempt_diagnostics or [],
            diagnostics_artifact_path=diagnostics_artifact_path,
            diagnostics_artifact_sha256=diagnostics_artifact_sha256,
            diagnostics_persistence_error=diagnostics_persistence_error,
            message=(
                "Provider-backed whole-asset structure evidence is required "
                "before component assignments"
            ),
        )
    if not evaluation.accepted:
        raise ProviderResponseConformanceTerminalError(
            evaluation,
            diagnostics_artifact_path=diagnostics_artifact_path,
            diagnostics_artifact_sha256=diagnostics_artifact_sha256,
            attempt_diagnostics=attempt_diagnostics,
            diagnostics_persistence_error=diagnostics_persistence_error,
        )


def whole_asset_structure_evaluation_from_dict(
    payload: Mapping[str, Any],
) -> WholeAssetStructureEvaluation:
    """Rehydrate an internal evaluation, rejecting malformed evidence."""
    try:
        accepted = payload.get("accepted")
        raw_reason_codes = payload.get("reason_codes")
        robot_type = payload.get("robot_type")
        dof = payload.get("dof")
        segment_names = payload.get("segment_names")
        source_prim_inventory = payload.get("source_prim_inventory")
        valid = bool(
            isinstance(accepted, bool)
            and isinstance(raw_reason_codes, list | tuple)
            and all(
                isinstance(reason, str) and reason in _WHOLE_ASSET_STRUCTURE_REASONS
                for reason in raw_reason_codes
            )
            and accepted is (not raw_reason_codes)
            and (robot_type is None or isinstance(robot_type, str))
            and (dof is None or (isinstance(dof, int) and not isinstance(dof, bool)))
            and isinstance(segment_names, list | tuple)
            and all(isinstance(name, str) and name.strip() for name in segment_names)
            and isinstance(source_prim_inventory, list | tuple)
            and all(isinstance(path, str) for path in source_prim_inventory)
        )
    except (TypeError, ValueError):
        valid = False

    if not valid:
        return WholeAssetStructureEvaluation(
            accepted=False,
            reason_codes=("missing_whole_asset_source_contract",),
            robot_type=None,
            dof=None,
            segment_names=(),
            source_prim_inventory=(),
        )

    return WholeAssetStructureEvaluation(
        accepted=cast(bool, accepted),
        reason_codes=tuple(
            cast(
                list[WholeAssetStructureReason] | tuple[WholeAssetStructureReason, ...],
                raw_reason_codes,
            )
        ),
        robot_type=cast(str | None, robot_type),
        dof=cast(int | None, dof),
        segment_names=tuple(cast(list[str] | tuple[str, ...], segment_names)),
        source_prim_inventory=tuple(
            cast(list[str] | tuple[str, ...], source_prim_inventory)
        ),
    )


__all__ = [
    "PROVIDER_RESPONSE_CHECKPOINT_CALLBACK_KEY",
    "ProviderAttemptDiagnostic",
    "ProviderAttemptJournal",
    "ProviderAttemptJournalSnapshot",
    "ProviderAttemptOutcome",
    "ProviderAttemptPersistence",
    "ProviderAttemptReceipt",
    "ProviderAttemptRecording",
    "ProviderAttemptRecorderError",
    "ProviderRequestKind",
    "ProviderResponseConformanceTerminalError",
    "RawProviderResponseEvidence",
    "Stage1ResponseEvaluation",
    "WholeAssetStructureEvaluation",
    "build_provider_attempt_diagnostic",
    "evaluate_exhausted_transport_terminal",
    "evaluate_stage1_response",
    "evaluate_whole_asset_structure",
    "invoke_provider_attempt_recorder",
    "load_provider_attempt_journal",
    "persist_provider_attempt_journal",
    "project_provider_attempt_persistence",
    "provider_attempt_persistence_error",
    "raw_provider_response_evidence",
    "require_provider_attempt_journal_persistence",
    "require_whole_asset_structure",
    "run_provider_call_with_journal",
    "whole_asset_structure_evaluation_from_dict",
]
