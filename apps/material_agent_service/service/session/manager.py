# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for pipeline executions."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from world_understanding.utils.artifacts import (
    ArtifactPathError,
    OpenArtifactFile,
    open_held_confined_artifact,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.session_paths import (
    confined_session_path,
    safe_listed_session_ids,
)

from ..artifact_lineage import artifact_is_valid, current_artifact_validity
from ..storage.base import (
    METADATA_KEY,
    JsonPreconditionError,
    SessionMetadataContentionError,
    SessionStore,
    VersionedJson,
)
from ..storage.local_store import LocalSessionStore

logger = logging.getLogger(__name__)

# Cancel signal key
CANCEL_KEY = ".cancel"

# Session IDs are server-generated UUID4 strings but are also accepted back from
# URL path parameters (e.g. GET /sessions/{id}/...), so they must be validated
# before reaching any code that builds a filesystem path or storage key from
# them. The pattern is intentionally case-insensitive to tolerate normal UUID
# casing variance; it still rejects `../`, `/`, empty, and non-hex inputs.
_SESSION_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_REGENERATION_CLAIM_KEY = "regeneration_claim"
_PUBLISHED_ARTIFACTS_KEY = "published_artifacts"
_PREDICTION_REPORT_PUBLICATION_KEY = "prediction_report_publication"
_TERMINAL_EVENTS_QUIESCED_KEY = "terminal_events_quiesced"
_FENCED_METADATA_FIELDS = frozenset(
    {
        _REGENERATION_CLAIM_KEY,
        _PUBLISHED_ARTIFACTS_KEY,
        _PREDICTION_REPORT_PUBLICATION_KEY,
        _TERMINAL_EVENTS_QUIESCED_KEY,
    }
)
_MAX_CAS_ATTEMPTS = 32


class RegenerationClaimConflictError(RuntimeError):
    """Raised when a regeneration claim loses or violates its durable CAS."""


@dataclass(frozen=True)
class RegenerationClaim:
    """Opaque fencing identity for one regeneration attempt."""

    generation: int
    token: str
    lease_expires_at: datetime

    @property
    def artifact_prefix(self) -> str:
        """Immutable store prefix reserved for artifacts from this attempt."""
        return f"runs/{self.generation}-{self.token}"

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> RegenerationClaim | None:
        """Parse the current claim identity from session metadata."""
        raw = metadata.get(_REGENERATION_CLAIM_KEY)
        if not isinstance(raw, Mapping):
            return None
        generation = raw.get("generation")
        token = raw.get("token")
        lease_expires_at = raw.get("lease_expires_at")
        if (
            not isinstance(generation, int)
            or generation < 1
            or not isinstance(token, str)
            or not token
            or not isinstance(lease_expires_at, str)
        ):
            return None
        try:
            parsed_expiry = _parse_utc_datetime(lease_expires_at)
        except ValueError:
            return None
        return cls(
            generation=generation,
            token=token,
            lease_expires_at=parsed_expiry,
        )


def _parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class InvalidSessionIdError(ValueError):
    """Raised when a session_id fails format validation.

    Subclasses ValueError so existing `except ValueError` / `pytest.raises(ValueError)`
    keeps working, but lets FastAPI's exception handler target just this class
    instead of swallowing every ValueError in the app.
    """


def _validate_session_id(session_id: str) -> str:
    """Validate that session_id has UUID shape; reject otherwise."""
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise InvalidSessionIdError(f"Invalid session_id: {session_id!r}")
    return session_id


class SessionManager:
    """Manages pipeline sessions and their artifacts.

    All storage operations are delegated to the configured SessionStore backend.
    If no store is provided, defaults to LocalSessionStore.
    All methods are async to support non-blocking I/O.
    """

    def __init__(
        self,
        storage_path: Path | str,
        ttl_hours: int = 24,
        store: SessionStore | None = None,
    ):
        """Initialize session manager.

        Args:
            storage_path: Base directory for local session storage (used as
                          default root for LocalSessionStore if no store provided)
            ttl_hours: Time-to-live for sessions in hours
            store: Storage backend (defaults to LocalSessionStore at storage_path)
        """
        self.storage_path = Path(storage_path)
        self.ttl_hours = ttl_hours
        self._update_locks: dict[str, asyncio.Lock] = {}

        # Default to LocalSessionStore if no store provided
        if store is None:
            self._store = LocalSessionStore(str(self.storage_path))
        else:
            self._store = store

        # Ensure storage directory exists (for local store compatibility)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def store(self) -> SessionStore:
        """Get the configured storage backend."""
        return self._store

    # ---------- Session Lifecycle ----------

    async def create_session(
        self, session_id: str, config: dict[str, Any] | None = None
    ) -> Path:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            config: Optional configuration dict
        """
        session_id = _validate_session_id(session_id)
        # Initialize session in store
        await self._store.init_session(session_id)

        # Create local directory structure (for backward compat with file-based ops)
        session_dir = self.get_session_dir(session_id)

        # Create directory structure
        (session_dir / "input").mkdir(parents=True, exist_ok=True)
        (session_dir / "materials").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "dataset").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "predictions").mkdir(parents=True, exist_ok=True)
        (session_dir / "preview").mkdir(parents=True, exist_ok=True)
        (session_dir / "output").mkdir(parents=True, exist_ok=True)

        # Initialize session metadata
        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": 3,
                "percent": 0,
                "estimated_remaining_seconds": None,
            },
            "preview_images": [],
            "can_cancel": True,
            "elapsed_seconds": 0,
            "config": config or {},
            "ttl_expires_at": (
                datetime.now(UTC) + timedelta(hours=self.ttl_hours)
            ).isoformat(),
        }

        # Save metadata via store
        await self._store.put_json(session_id, METADATA_KEY, metadata)

        logger.info(f"Created session: {session_id}")
        return session_dir

    def get_session_dir(self, session_id: str) -> Path:
        """Get path to local session directory.

        Args:
            session_id: Session identifier

        Returns:
            Path to session directory
        """
        return confined_session_path(
            self.storage_path,
            _validate_session_id(session_id),
        )

    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session exists
        """
        session_id = _validate_session_id(session_id)
        return await self._store.exists(session_id, METADATA_KEY)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        session_id = _validate_session_id(session_id)
        try:
            await self._store.delete_session(session_id)
            self._update_locks.pop(session_id, None)
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception:
            log_durable_failure(
                logger,
                "session_store_delete_failed",
                phase=FailurePhase.ROLLBACK,
                retryable=True,
            )
            return False

    # ---------- Metadata Operations ----------

    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Session metadata dict or None if not found
        """
        session_id = _validate_session_id(session_id)
        return await self._store.get_json(session_id, METADATA_KEY)

    async def get_session_metadata_versioned(self, session_id: str) -> VersionedJson:
        """Read metadata together with its opaque conditional-write version."""
        session_id = _validate_session_id(session_id)
        return await self._store.get_json_versioned(session_id, METADATA_KEY)

    async def get_session_metadata_batch(
        self, session_ids: list[str]
    ) -> list[dict[str, Any] | None]:
        """Get metadata for multiple sessions in a single batch.

        Uses the store's batch method to reuse a single connection.

        Args:
            session_ids: List of session identifiers

        Returns:
            List of metadata dicts (or None), matching input order
        """
        session_ids = [_validate_session_id(sid) for sid in session_ids]
        return await self._store.get_json_batch(session_ids, METADATA_KEY)

    def _get_update_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock for update_session.

        Prevents concurrent read-modify-write races when multiple callers
        (e.g., EventBus and executor) update the same session concurrently.
        """
        return self._update_locks.setdefault(session_id, asyncio.Lock())

    @staticmethod
    def _apply_metadata_updates(
        metadata: dict[str, Any],
        updates: Mapping[str, Any],
        remove_fields: Iterable[str],
        *,
        now: datetime,
    ) -> None:
        for field in remove_fields:
            metadata.pop(field, None)
        metadata.update(updates)
        metadata["updated_at"] = now.isoformat()

        created_at_value = metadata.get("created_at")
        if isinstance(created_at_value, str):
            try:
                created_at = _parse_utc_datetime(created_at_value)
            except ValueError:
                return
            metadata["elapsed_seconds"] = int((now - created_at).total_seconds())

    @staticmethod
    def _validate_fenced_updates(
        updates: Mapping[str, Any], remove_fields: Iterable[str]
    ) -> tuple[str, ...]:
        removals = tuple(remove_fields)
        reserved = (_FENCED_METADATA_FIELDS & updates.keys()) | (
            _FENCED_METADATA_FIELDS & set(removals)
        )
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"Fenced metadata fields cannot be updated directly: {names}"
            )
        return removals

    @staticmethod
    def _claim_matches(
        metadata: Mapping[str, Any],
        claim: RegenerationClaim,
        *,
        now: datetime,
        require_active_lease: bool = True,
    ) -> bool:
        raw = metadata.get(_REGENERATION_CLAIM_KEY)
        current = RegenerationClaim.from_metadata(metadata)
        if not isinstance(raw, Mapping) or current is None:
            return False
        if current.generation != claim.generation or current.token != claim.token:
            return False
        if require_active_lease:
            return raw.get("active") is True and current.lease_expires_at > now
        return True

    @staticmethod
    def _has_active_regeneration_claim(metadata: Mapping[str, Any]) -> bool:
        """Return whether metadata is owned by any regeneration worker.

        An expired claim remains fenced until the explicit takeover or expired-
        cancellation CAS resolves it.  Letting ordinary writers mutate in that
        window would reintroduce the same cross-instance race the claim prevents.
        """
        raw_claim = metadata.get(_REGENERATION_CLAIM_KEY)
        return isinstance(raw_claim, Mapping) and raw_claim.get("active") is True

    @classmethod
    def _reject_active_regeneration(cls, metadata: Mapping[str, Any]) -> None:
        if cls._has_active_regeneration_claim(metadata):
            raise RegenerationClaimConflictError(
                "Session is fenced by an active regeneration claim"
            )

    async def update_session(
        self,
        session_id: str,
        updates: dict[str, Any],
        *,
        remove_fields: Iterable[str] = (),
        sync_files: bool = True,
    ) -> None:
        """Update session metadata.

        Args:
            session_id: Session identifier
            updates: Dictionary of fields to update
            remove_fields: Existing fields to remove before applying updates
            sync_files: Whether to mirror local artifacts after metadata is saved
        """
        session_id = _validate_session_id(session_id)
        removals = self._validate_fenced_updates(updates, remove_fields)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    logger.warning(f"Cannot update non-existent session: {session_id}")
                    return

                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                self._apply_metadata_updates(
                    metadata,
                    updates,
                    removals,
                    now=datetime.now(UTC),
                )
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    break
                except JsonPreconditionError:
                    continue
            else:
                raise SessionMetadataContentionError(
                    f"Session metadata remained contended for {session_id}"
                )

            # sync session to store
            if sync_files:
                await self.sync_session_to_store(session_id)

    async def finalize_standard_pipeline(
        self,
        session_id: str,
        updates: Mapping[str, Any],
        *,
        remove_fields: Iterable[str] = (),
    ) -> bool:
        """Persist standard terminal metadata before its final EventBus emit.

        Regeneration admission remains closed until the executor emits (or
        deliberately skips) its terminal event and calls
        :meth:`mark_terminal_events_quiesced`.
        """
        session_id = _validate_session_id(session_id)
        removals = self._validate_fenced_updates(updates, remove_fields)
        if updates.get("status") not in {"completed", "failed", "cancelled"}:
            raise ValueError("Standard pipeline finalization requires terminal status")
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                cancellation_won = metadata.get("status") in {
                    "cancelling",
                    "cancelled",
                } or await self._store.exists(session_id, CANCEL_KEY)
                if cancellation_won and updates.get("status") != "cancelled":
                    return False
                self._apply_metadata_updates(
                    metadata,
                    updates,
                    removals,
                    now=datetime.now(UTC),
                )
                metadata[_TERMINAL_EVENTS_QUIESCED_KEY] = False
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def mark_terminal_events_quiesced(
        self,
        session_id: str,
        *,
        expected_status: str,
    ) -> bool:
        """Open regeneration admission after the standard terminal emit phase."""
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                if metadata.get("status") != expected_status:
                    return False
                if metadata.get(_TERMINAL_EVENTS_QUIESCED_KEY) is True:
                    return True
                if metadata.get(_TERMINAL_EVENTS_QUIESCED_KEY) is not False:
                    return False
                metadata[_TERMINAL_EVENTS_QUIESCED_KEY] = True
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def claim_regeneration(
        self,
        session_id: str,
        *,
        expected_version: str,
        updates: Mapping[str, Any] | None = None,
        remove_fields: Iterable[str] = (),
        lease_seconds: float = 300.0,
    ) -> RegenerationClaim:
        """Atomically fence one regeneration attempt.

        The caller must pass the metadata version used to plan the run.  Two
        service instances planning from the same version can both reach this
        method, but exactly one conditional replacement can win.  An expired
        active lease may be taken over only through the same CAS operation.
        """
        session_id = _validate_session_id(session_id)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        requested_updates = dict(updates or {})
        removals = self._validate_fenced_updates(requested_updates, remove_fields)

        async with self._get_update_lock(session_id):
            snapshot = await self.get_session_metadata_versioned(session_id)
            if snapshot.value is None or snapshot.version is None:
                raise RegenerationClaimConflictError("Session not found")
            if snapshot.version != expected_version:
                raise RegenerationClaimConflictError(
                    "Session changed while regeneration was being planned"
                )

            metadata = dict(snapshot.value)
            now = datetime.now(UTC)
            raw_claim = metadata.get(_REGENERATION_CLAIM_KEY)
            current_claim = RegenerationClaim.from_metadata(metadata)
            if isinstance(raw_claim, Mapping) and raw_claim.get("active") is True:
                if current_claim is None:
                    raise RegenerationClaimConflictError(
                        "Session has an invalid active regeneration lease"
                    )
                if current_claim.lease_expires_at > now:
                    raise RegenerationClaimConflictError(
                        "A regeneration lease is already active"
                    )
            if metadata.get(_TERMINAL_EVENTS_QUIESCED_KEY) is False:
                raise RegenerationClaimConflictError(
                    "Pipeline terminal events are still being finalized"
                )

            expired_takeover = (
                isinstance(raw_claim, Mapping)
                and raw_claim.get("active") is True
                and current_claim is not None
                and current_claim.lease_expires_at <= now
            )
            if (
                metadata.get("status") not in {"completed", "failed", "cancelled"}
                and not expired_takeover
            ):
                raise RegenerationClaimConflictError(
                    f"Cannot regenerate while pipeline is {metadata.get('status')}"
                )

            prior_generations: list[int] = []
            for value in (raw_claim, metadata.get(_PUBLISHED_ARTIFACTS_KEY)):
                if isinstance(value, Mapping):
                    generation = value.get("generation")
                    if isinstance(generation, int) and generation >= 0:
                        prior_generations.append(generation)
            generation = max(prior_generations, default=0) + 1
            token = str(uuid4())
            lease_expires_at = now + timedelta(seconds=lease_seconds)
            claim = RegenerationClaim(
                generation=generation,
                token=token,
                lease_expires_at=lease_expires_at,
            )
            requested_updates.setdefault("status", "pending")
            self._apply_metadata_updates(
                metadata,
                requested_updates,
                removals,
                now=now,
            )
            metadata[_REGENERATION_CLAIM_KEY] = {
                "generation": generation,
                "token": token,
                "active": True,
                "claimed_at": now.isoformat(),
                "renewed_at": now.isoformat(),
                "lease_expires_at": lease_expires_at.isoformat(),
                "cancel_requested": False,
            }
            try:
                await self._store.replace_json_if_version(
                    session_id,
                    METADATA_KEY,
                    metadata,
                    expected_version,
                )
            except JsonPreconditionError as exc:
                raise RegenerationClaimConflictError(
                    "Another regeneration attempt won the session claim"
                ) from exc
            return claim

    async def update_session_for_claim(
        self,
        session_id: str,
        claim: RegenerationClaim,
        updates: Mapping[str, Any],
        *,
        remove_fields: Iterable[str] = (),
    ) -> bool:
        """Conditionally update metadata while ``claim`` owns a live lease.

        Returns ``False`` for an expired, superseded, or finalized claim.  In
        particular, a stale worker finishing after a takeover cannot update
        durable status or artifact publication metadata.
        """
        session_id = _validate_session_id(session_id)
        removals = self._validate_fenced_updates(updates, remove_fields)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                now = datetime.now(UTC)
                if not self._claim_matches(metadata, claim, now=now):
                    return False
                self._apply_metadata_updates(
                    metadata,
                    updates,
                    removals,
                    now=now,
                )
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def renew_regeneration_claim(
        self,
        session_id: str,
        claim: RegenerationClaim,
        *,
        lease_seconds: float = 300.0,
    ) -> bool:
        """Extend a live lease without allowing an expired lease to revive."""
        session_id = _validate_session_id(session_id)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                now = datetime.now(UTC)
                if not self._claim_matches(metadata, claim, now=now):
                    return False
                raw_claim = dict(metadata[_REGENERATION_CLAIM_KEY])
                lease_expires_at = now + timedelta(seconds=lease_seconds)
                raw_claim["renewed_at"] = now.isoformat()
                raw_claim["lease_expires_at"] = lease_expires_at.isoformat()
                metadata[_REGENERATION_CLAIM_KEY] = raw_claim
                metadata["updated_at"] = now.isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def cancel_regeneration_claim(
        self,
        session_id: str,
        claim: RegenerationClaim,
    ) -> bool:
        """Atomically request cancellation for exactly one claimed generation."""
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                now = datetime.now(UTC)
                if not self._claim_matches(metadata, claim, now=now):
                    return False
                raw_claim = dict(metadata[_REGENERATION_CLAIM_KEY])
                raw_claim["cancel_requested"] = True
                raw_claim["cancel_requested_at"] = now.isoformat()
                metadata[_REGENERATION_CLAIM_KEY] = raw_claim
                self._apply_metadata_updates(
                    metadata,
                    {"status": "cancelling"},
                    (),
                    now=now,
                )
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def cancel_expired_regeneration_claim(
        self,
        session_id: str,
        claim: RegenerationClaim,
    ) -> bool:
        """CAS-cancel an expired claim if no takeover has won yet.

        A concurrent takeover and this recovery both condition on the same
        metadata version, so exactly one can win.  This method never cancels a
        live lease or a different generation/token.
        """
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                now = datetime.now(UTC)
                raw_claim = metadata.get(_REGENERATION_CLAIM_KEY)
                current_claim = RegenerationClaim.from_metadata(metadata)
                if (
                    not isinstance(raw_claim, Mapping)
                    or raw_claim.get("active") is not True
                    or current_claim is None
                    or current_claim.generation != claim.generation
                    or current_claim.token != claim.token
                    or current_claim.lease_expires_at > now
                ):
                    return False

                updated_claim = dict(raw_claim)
                updated_claim["active"] = False
                updated_claim["cancel_requested"] = True
                updated_claim["cancel_requested_at"] = now.isoformat()
                updated_claim["finalized_at"] = now.isoformat()
                metadata[_REGENERATION_CLAIM_KEY] = updated_claim
                self._apply_metadata_updates(
                    metadata,
                    {
                        "status": "cancelled",
                        "current_step": None,
                        "can_cancel": False,
                        "cancelled_at": now.isoformat(),
                    },
                    (),
                    now=now,
                )
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def is_regeneration_cancel_requested(
        self,
        session_id: str,
        claim: RegenerationClaim,
    ) -> bool:
        """Return whether cancellation was requested for the current claim."""
        session_id = _validate_session_id(session_id)
        metadata = await self.get_session_metadata(session_id)
        if metadata is None or not self._claim_matches(
            metadata,
            claim,
            now=datetime.now(UTC),
        ):
            return False
        raw_claim = metadata.get(_REGENERATION_CLAIM_KEY)
        return (
            isinstance(raw_claim, Mapping) and raw_claim.get("cancel_requested") is True
        )

    async def finalize_regeneration_claim(
        self,
        session_id: str,
        claim: RegenerationClaim,
        *,
        updates: Mapping[str, Any],
        artifact_map: Mapping[str, str] | None = None,
        remove_fields: Iterable[str] = (),
    ) -> bool:
        """Atomically publish terminal metadata and immutable artifact pointers."""
        session_id = _validate_session_id(session_id)
        removals = self._validate_fenced_updates(updates, remove_fields)
        final_updates = dict(updates)
        immutable_artifacts = dict(artifact_map) if artifact_map is not None else None
        if immutable_artifacts is not None:
            expected_prefix = f"{claim.artifact_prefix}/"
            for logical_name, artifact_key in immutable_artifacts.items():
                if (
                    not isinstance(logical_name, str)
                    or not logical_name
                    or not isinstance(artifact_key, str)
                ):
                    raise ValueError(
                        "Artifact publication entries must be named strings"
                    )
                if not artifact_key.startswith(expected_prefix):
                    raise ValueError(
                        "Regeneration artifacts must use the claim's immutable prefix"
                    )
                if not await self._store.exists(session_id, artifact_key):
                    raise FileNotFoundError(
                        f"Cannot publish missing regeneration artifact: {artifact_key}"
                    )

        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                now = datetime.now(UTC)
                if not self._claim_matches(metadata, claim, now=now):
                    return False
                current_claim = metadata.get(_REGENERATION_CLAIM_KEY)
                if (
                    isinstance(current_claim, Mapping)
                    and current_claim.get("cancel_requested") is True
                    and final_updates.get("status") != "cancelled"
                ):
                    # Cancellation is a durable ownership decision.  A worker
                    # that finishes between the cancel CAS and task delivery
                    # must not publish completed/failed terminal metadata.
                    return False
                requested_validity = final_updates.get("artifact_validity")
                current_publication = metadata.get(_PREDICTION_REPORT_PUBLICATION_KEY)
                resulting_lineage = final_updates.get(
                    "prediction_lineage_token",
                    metadata.get("prediction_lineage_token"),
                )
                if (
                    isinstance(requested_validity, Mapping)
                    and isinstance(current_publication, Mapping)
                    and current_publication.get("prediction_lineage_token")
                    == resulting_lineage
                    and isinstance(current_publication.get("key"), str)
                    and artifact_is_valid(metadata, "prediction_report")
                    and requested_validity.get("raw_predictions") is True
                ):
                    # The executor derives its validity map before final CAS.
                    # Preserve a report published for the unchanged prediction
                    # lineage in the meantime instead of writing that bit back
                    # to false while retaining its immutable pointer.
                    merged_validity = dict(requested_validity)
                    merged_validity["prediction_report"] = True
                    final_updates["artifact_validity"] = merged_validity
                self._apply_metadata_updates(
                    metadata,
                    final_updates,
                    removals,
                    now=now,
                )
                raw_claim = dict(metadata[_REGENERATION_CLAIM_KEY])
                raw_claim["active"] = False
                raw_claim["finalized_at"] = now.isoformat()
                metadata[_REGENERATION_CLAIM_KEY] = raw_claim
                if immutable_artifacts is not None:
                    metadata[_PUBLISHED_ARTIFACTS_KEY] = {
                        "generation": claim.generation,
                        "token": claim.token,
                        "prefix": claim.artifact_prefix,
                        "artifacts": immutable_artifacts,
                        "published_at": now.isoformat(),
                    }
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def abort_regeneration_claim(
        self,
        session_id: str,
        claim: RegenerationClaim,
        *,
        restore_metadata: Mapping[str, Any],
    ) -> bool:
        """CAS-restore a pre-claim snapshot after preparation fails.

        The prior metadata is restored exactly apart from timestamps and an
        inactive record of the aborted generation.  Keeping that generation
        record prevents a later claim from reusing its immutable run prefix.
        A superseded or expired worker cannot roll back a newer owner.
        """
        session_id = _validate_session_id(session_id)
        restored_template = dict(restore_metadata)
        restored_session_id = restored_template.get("session_id")
        if restored_session_id != session_id:
            raise ValueError("restore_metadata belongs to a different session")

        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                current_metadata = dict(snapshot.value)
                now = datetime.now(UTC)
                if not self._claim_matches(current_metadata, claim, now=now):
                    return False

                raw_claim = dict(current_metadata[_REGENERATION_CLAIM_KEY])
                raw_claim["active"] = False
                raw_claim["aborted_at"] = now.isoformat()
                restored = dict(restored_template)
                restored[_REGENERATION_CLAIM_KEY] = raw_claim
                restored["updated_at"] = now.isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        restored,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    @staticmethod
    def resolve_published_artifact_key(
        metadata: Mapping[str, Any],
        logical_name: str,
        *,
        legacy_key: str | None = None,
    ) -> str | None:
        """Resolve a logical artifact without falling back across generations.

        Once a session has a publication map, a missing or malformed entry
        fails closed instead of falling back to mutable canonical bytes.  The
        optional legacy key is used only for sessions with no publication map.
        """
        publication = metadata.get(_PUBLISHED_ARTIFACTS_KEY)
        if publication is None:
            return legacy_key
        if not isinstance(publication, Mapping):
            return None
        prefix = publication.get("prefix")
        artifacts = publication.get("artifacts")
        if not isinstance(prefix, str) or not isinstance(artifacts, Mapping):
            return None
        artifact_key = artifacts.get(logical_name)
        if not isinstance(artifact_key, str):
            return None
        if not artifact_key.startswith(f"{prefix}/"):
            return None
        return artifact_key

    @staticmethod
    def resolve_prediction_report_key(
        metadata: Mapping[str, Any],
        *,
        legacy_key: str | None = "cache/predictions/prediction_report.html",
    ) -> str | None:
        """Resolve the report pointer only for the active prediction lineage."""
        publication = metadata.get(_PREDICTION_REPORT_PUBLICATION_KEY)
        if publication is None:
            return legacy_key
        if not isinstance(publication, Mapping):
            return None
        lineage = metadata.get("prediction_lineage_token")
        if publication.get("prediction_lineage_token") != lineage:
            return None
        key = publication.get("key")
        return key if isinstance(key, str) and key else None

    async def capture_prediction_lineage(self, session_id: str) -> str | None:
        """Return a stable current-prediction token, creating one for legacy data.

        The validity check and token creation are serialized with all metadata
        updates so on-demand report generation cannot start from invalidated
        prediction inputs.
        """
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return None
                metadata = dict(snapshot.value)
                if not artifact_is_valid(metadata, "raw_predictions"):
                    return None

                token = metadata.get("prediction_lineage_token")
                if isinstance(token, str) and token:
                    return token

                token = str(uuid4())
                metadata["prediction_lineage_token"] = token
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return token
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def publish_prediction_report_if_lineage_matches(
        self,
        session_id: str,
        expected_prediction_lineage: str,
        immutable_report_key: str,
    ) -> bool:
        """CAS-publish an already-uploaded immutable report object.

        Report bytes must be written before this call under
        ``reports/{lineage}/{publication-id}/...``.  This method publishes only
        the pointer and validity bit; it never replaces a canonical report file.
        """
        session_id = _validate_session_id(session_id)
        if not expected_prediction_lineage or "/" in expected_prediction_lineage:
            raise ValueError("Prediction lineage must be a non-empty path segment")
        expected_prefix = f"reports/{expected_prediction_lineage}/"
        relative_key = immutable_report_key.removeprefix(expected_prefix)
        if (
            not immutable_report_key.startswith(expected_prefix)
            or "/" not in relative_key
            or immutable_report_key.endswith("/")
        ):
            raise ValueError(
                "Prediction reports must use an immutable lineage/publication key"
            )
        if not await self._store.exists(session_id, immutable_report_key):
            return False

        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                if metadata.get(
                    "prediction_lineage_token"
                ) != expected_prediction_lineage or not artifact_is_valid(
                    metadata, "raw_predictions"
                ):
                    return False

                now = datetime.now(UTC)
                validity = current_artifact_validity(metadata)
                validity["prediction_report"] = True
                metadata["artifact_validity"] = validity
                metadata[_PREDICTION_REPORT_PUBLICATION_KEY] = {
                    "prediction_lineage_token": expected_prediction_lineage,
                    "key": immutable_report_key,
                    "published_at": now.isoformat(),
                }
                metadata["updated_at"] = now.isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return True
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def mark_prediction_report_valid_if_lineage_matches(
        self,
        session_id: str,
        expected_prediction_lineage: str,
        report_path: Path,
        *,
        report_key: str = "cache/predictions/prediction_report.html",
    ) -> bool:
        """Upload a report immutably, then CAS-publish its lineage pointer.

        ``report_key`` remains accepted for API compatibility but is used only
        as the immutable object's basename.  Canonical report bytes are never
        replaced by this method.
        """
        session_id = _validate_session_id(session_id)
        basename = Path(report_key).name
        immutable_key = f"reports/{expected_prediction_lineage}/{uuid4()}/{basename}"
        await self._store.put_file(
            session_id,
            immutable_key,
            str(report_path),
            "text/html",
        )
        return await self.publish_prediction_report_if_lineage_matches(
            session_id,
            expected_prediction_lineage,
            immutable_key,
        )

    async def update_step_progress(
        self,
        session_id: str,
        step_name: str,
        progress: dict[str, Any],
    ) -> None:
        """Update progress for current step.

        Args:
            session_id: Session identifier
            step_name: Name of current step
            progress: Progress dict with current, total, percent, message
        """
        # Map step names to display names and step numbers
        step_info_map = {
            "build_dataset_usd": {"display": "Rendering USD Scene", "step_num": 1},
            "predict": {"display": "Running VLM Predictions", "step_num": 2},
            "apply": {"display": "Applying Materials", "step_num": 3},
        }

        step_info = step_info_map.get(step_name, {"display": step_name, "step_num": 0})
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                now = datetime.now(UTC)

                current_step = metadata.get("current_step")
                if (
                    isinstance(current_step, Mapping)
                    and current_step.get("name") == step_name
                ):
                    current_step_info = dict(current_step)
                    started_at = datetime.fromisoformat(
                        str(current_step_info["started_at"])
                    )
                    current_step_info["progress"] = progress
                    current_step_info["elapsed_seconds"] = int(
                        (now - started_at).total_seconds()
                    )
                else:
                    current_step_info = {
                        "name": step_name,
                        "display_name": step_info["display"],
                        "started_at": now.isoformat(),
                        "progress": progress,
                        "elapsed_seconds": 0,
                    }
                metadata["current_step"] = current_step_info

                step_num = step_info["step_num"]
                if step_num > 0:
                    overall_progress = dict(metadata.get("overall_progress", {}))
                    overall_progress["current_step"] = step_num
                    overall_progress["percent"] = min(100, progress.get("percent", 0))
                    metadata["overall_progress"] = overall_progress
                metadata["updated_at"] = now.isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def mark_step_completed(
        self,
        session_id: str,
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Mark a step as completed.

        Args:
            session_id: Session identifier
            step_name: Name of completed step
            stats: Optional statistics from step execution
        """
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                current_step_info = metadata.get("current_step")
                if not (
                    isinstance(current_step_info, Mapping)
                    and current_step_info.get("name") == step_name
                ):
                    return

                completed_at = datetime.now(UTC)
                started_at = datetime.fromisoformat(
                    str(current_step_info["started_at"])
                )
                duration = int((completed_at - started_at).total_seconds())
                completed_step = {
                    "name": step_name,
                    "display_name": current_step_info["display_name"],
                    "started_at": current_step_info["started_at"],
                    "completed_at": completed_at.isoformat(),
                    "duration_seconds": duration,
                    "stats": stats or {},
                }
                completed_steps = list(metadata.get("completed_steps", []))
                completed_steps.append(completed_step)
                metadata["completed_steps"] = completed_steps
                timings = dict(metadata.get("timings", {}))
                timings[step_name] = duration
                metadata["timings"] = timings
                metadata["current_step"] = None

                completed_count = len(completed_steps)
                overall_progress = dict(metadata.get("overall_progress", {}))
                overall_progress["current_step"] = completed_count
                cumulative_percents = [50, 90, 100]
                overall_progress["percent"] = (
                    cumulative_percents[completed_count - 1]
                    if completed_count <= len(cumulative_percents)
                    else 100
                )
                metadata["overall_progress"] = overall_progress
                metadata["updated_at"] = completed_at.isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def add_preview_image(self, session_id: str, image_name: str) -> None:
        """Add a preview image to the session.

        Args:
            session_id: Session identifier
            image_name: Name of preview image file
        """
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                preview_images = list(metadata.get("preview_images", []))
                if image_name in preview_images:
                    return
                preview_images.append(image_name)
                metadata["preview_images"] = preview_images
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def update_preview_images(
        self, session_id: str, image_names: list[str]
    ) -> None:
        """Update the list of preview images.

        Args:
            session_id: Session identifier
            image_names: List of preview image filenames
        """
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                metadata["preview_images"] = list(image_names)
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    return
                except JsonPreconditionError:
                    continue
        raise SessionMetadataContentionError(
            f"Session metadata remained contended for {session_id}"
        )

    async def add_generated_reference_image(
        self, session_id: str, entry: dict[str, Any]
    ) -> bool:
        """CAS-append a generated-reference record while the session is ready."""
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return False
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                if metadata.get("status") != "ready":
                    raise RegenerationClaimConflictError(
                        "Generated references can only be changed while the session is ready"
                    )
                generated_refs = list(metadata.get("generated_reference_images", []))
                generated_refs.append(dict(entry))
                metadata["generated_reference_images"] = generated_refs
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    break
                except JsonPreconditionError:
                    continue
            else:
                raise SessionMetadataContentionError(
                    f"Session metadata remained contended for {session_id}"
                )
        await self.sync_session_to_store(session_id)
        return True

    async def remove_generated_reference_image(
        self, session_id: str, reference_id: str
    ) -> dict[str, Any] | None:
        """Remove a generated-reference metadata record by ID."""
        session_id = _validate_session_id(session_id)
        async with self._get_update_lock(session_id):
            for _attempt in range(_MAX_CAS_ATTEMPTS):
                snapshot = await self.get_session_metadata_versioned(session_id)
                if snapshot.value is None or snapshot.version is None:
                    return None
                metadata = dict(snapshot.value)
                self._reject_active_regeneration(metadata)
                if metadata.get("status") != "ready":
                    raise RegenerationClaimConflictError(
                        "Generated references can only be changed while the session is ready"
                    )
                generated_refs = list(metadata.get("generated_reference_images", []))
                kept_refs = [
                    ref for ref in generated_refs if ref.get("id") != reference_id
                ]
                if len(kept_refs) == len(generated_refs):
                    return None
                removed_ref = next(
                    ref for ref in generated_refs if ref.get("id") == reference_id
                )
                metadata["generated_reference_images"] = kept_refs
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                try:
                    await self._store.replace_json_if_version(
                        session_id,
                        METADATA_KEY,
                        metadata,
                        snapshot.version,
                    )
                    break
                except JsonPreconditionError:
                    continue
            else:
                raise SessionMetadataContentionError(
                    f"Session metadata remained contended for {session_id}"
                )
        await self.sync_session_to_store(session_id)
        return removed_ref

    # ---------- Cancellation ----------

    async def is_cancelled(self, session_id: str) -> bool:
        """Check if session has been cancelled.

        Args:
            session_id: Session identifier

        Returns:
            True if cancellation signal exists
        """
        session_id = _validate_session_id(session_id)
        return await self._store.exists(session_id, CANCEL_KEY)

    async def request_cancellation(self, session_id: str) -> None:
        """Request cancellation of a running pipeline.

        Args:
            session_id: Session identifier
        """
        session_id = _validate_session_id(session_id)
        if not await self.session_exists(session_id):
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return

        # Create cancellation signal
        await self._store.put_bytes(session_id, CANCEL_KEY, b"")

        # Update status
        await self.update_session(session_id, {"status": "cancelling"})

        logger.info(f"Cancellation requested for session: {session_id}")

    async def clear_cancellation(self, session_id: str) -> None:
        """Remove durable and local cancellation markers before a new run."""
        session_id = _validate_session_id(session_id)
        await self._store.delete_file(session_id, CANCEL_KEY)
        (self.get_session_dir(session_id) / CANCEL_KEY).unlink(missing_ok=True)

    async def restore_cancellation(
        self,
        session_id: str,
        *,
        cancelled: bool,
    ) -> None:
        """Restore cancellation-marker presence during transactional rollback."""
        session_id = _validate_session_id(session_id)
        if cancelled:
            await self._store.put_bytes(session_id, CANCEL_KEY, b"")
            if self._store.kind != "local":
                (self.get_session_dir(session_id) / CANCEL_KEY).write_bytes(b"")
        else:
            await self.clear_cancellation(session_id)

    # ---------- Artifact Operations ----------

    async def get_artifact_path(
        self, session_id: str, artifact_type: str
    ) -> Path | None:
        """Get path to a session artifact (local filesystem).

        Args:
            session_id: Session identifier
            artifact_type: Type of artifact (output_usd, predictions, etc.)

        Returns:
            Path to artifact or None if not found
        """
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)

        artifact_map = {
            "output_usd": session_dir / "output" / "scene_with_materials.usd",
            "predictions": session_dir / "cache" / "predictions" / "predictions.jsonl",
            "dataset": session_dir / "cache" / "dataset" / "dataset.jsonl",
        }

        path = artifact_map.get(artifact_type)
        if path:
            artifact = await self.open_local_artifact(session_id, path)
            if artifact is not None:
                artifact.stream.close()
                return path

        return None

    async def open_local_artifact(
        self,
        session_id: str,
        local_path: Path,
    ) -> OpenArtifactFile | None:
        """Open a session-owned local artifact through a held descriptor chain."""

        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        try:
            return await asyncio.to_thread(
                open_held_confined_artifact,
                session_dir,
                local_path,
            )
        except (ArtifactPathError, OSError, RuntimeError, ValueError):
            return None

    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None:
        """Generate a presigned/public URL for an artifact if store supports it.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/input_render.png")
            expires_seconds: URL expiration time

        Returns:
            Presigned URL string or None if not supported
        """
        session_id = _validate_session_id(session_id)
        return await self._store.make_public_url(session_id, key, expires_seconds)

    async def put_file_to_store(
        self,
        session_id: str,
        key: str,
        file_path: str,
        content_type: str | None = None,
    ) -> None:
        """Copy a file to the store.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/scene.usd")
            file_path: Local file path to copy
            content_type: Optional MIME type
        """
        session_id = _validate_session_id(session_id)
        await self._store.put_file(session_id, key, file_path, content_type)

    async def put_bytes_to_store(
        self,
        session_id: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        """Write bytes to the store.

        Args:
            session_id: Session identifier
            key: Artifact key
            data: Bytes to write
            content_type: Optional MIME type
        """
        session_id = _validate_session_id(session_id)
        await self._store.put_bytes(session_id, key, data, content_type)

    async def exists_in_store(self, session_id: str, key: str) -> bool:
        """Check if a file exists in the store.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/input_render.png")

        Returns:
            True if file exists in store
        """
        session_id = _validate_session_id(session_id)
        return await self._store.exists(session_id, key)

    async def read_from_store(self, session_id: str, key: str) -> bytes | None:
        """Read file content from the store.

        Args:
            session_id: Session identifier
            key: Artifact key (e.g., "input/input_render.png")

        Returns:
            File content as bytes, or None if not found
        """
        session_id = _validate_session_id(session_id)
        try:
            if not await self._store.exists(session_id, key):
                return None
            stream = await self._store.open_read(session_id, key)
            return stream.read()
        except Exception:
            log_durable_failure(
                logger,
                "session_store_read_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
            return None

    async def iter_store_chunks(
        self,
        session_id: str,
        key: str,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes] | None:
        """Return a lazy bounded store iterator, or ``None`` when absent."""
        session_id = _validate_session_id(session_id)
        try:
            if not await self._store.exists(session_id, key):
                return None
            return self._store.iter_read(
                session_id,
                key,
                chunk_size=chunk_size,
            )
        except Exception:
            log_durable_failure(
                logger,
                "session_store_stream_prepare_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
            return None

    # ---------- Sync ----------

    async def sync_session_to_store(self, session_id: str, prefix: str = "") -> int:
        """Sync all local session files to the remote store.

        For local storage, this is a no-op.
        For S3 storage, uploads all local files to S3.

        Args:
            session_id: Session identifier
            prefix: Optional prefix to filter files (e.g., "output/")

        Returns:
            Number of files synced
        """
        session_id = _validate_session_id(session_id)
        local_dir = str(self.get_session_dir(session_id))
        count = await self._store.sync_from_local(session_id, local_dir, prefix)
        if count > 0:
            logger.info(f"Synced {count} files to store for session {session_id[:8]}")
        return count

    async def sync_from_store(self, session_id: str, prefix: str = "") -> int:
        """Pull files from the store to local session directory.

        For local storage, this is a no-op.
        For S3 storage, downloads files from S3 to local disk.

        Args:
            session_id: Session identifier
            prefix: Optional prefix to filter files (e.g., "input/")

        Returns:
            Number of files downloaded
        """
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        count = await self._store.sync_to_local(
            session_id, str(session_dir), prefix=prefix
        )
        if count > 0:
            logger.info(f"Pulled {count} files from store for session {session_id[:8]}")
        return count

    # ---------- Session Listing & Cleanup ----------

    async def list_sessions(self) -> list[str]:
        """List all session IDs.

        Delegates to the configured storage backend to list sessions.
        For S3 storage, sessions are listed from the remote bucket.
        For local storage, sessions are listed from the local directory.

        Returns:
            List of session IDs
        """
        return safe_listed_session_ids(await self._store.list_sessions())

    async def cleanup_expired_sessions(self) -> int:
        """Remove sessions past their TTL.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0
        now = datetime.now(UTC)

        for session_id in await self.list_sessions():
            metadata = await self.get_session_metadata(session_id)

            if not metadata:
                continue

            if self.store.kind == "local" and self.ttl_hours > 0:
                expires_at_str = metadata.get("ttl_expires_at")
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str.replace("Z", ""))
                    if now > expires_at:
                        logger.info(f"Cleaning up expired session: {session_id}")
                        if await self.delete_session(session_id):
                            cleaned += 1
            else:
                logger.info(
                    "Skipping cleanup of session: %s (not local or TTL not enabled)",
                    session_id,
                )

        return cleaned

    async def cleanup_stale_local_cache(self, max_age_hours: float = 24.0) -> int:
        """Clean up stale local session cache.

        For S3 storage, syncs old sessions to remote and removes local files
        to free up disk space. Sessions not updated for longer than max_age_hours
        are considered stale.

        For local storage, this is a no-op since files are already in their
        final location.

        Args:
            max_age_hours: Maximum age in hours before cleanup (default: 24)

        Returns:
            Number of sessions cleaned up
        """
        return await self._store.cleanup_stale_local_sessions(
            str(self.storage_path), max_age_hours
        )
