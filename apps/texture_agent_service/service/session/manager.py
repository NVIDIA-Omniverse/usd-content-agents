# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for texture agent pipeline executions."""

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

from botocore.exceptions import BotoCoreError, ClientError
from filelock import FileLock, Timeout

from ..storage import (
    CANCEL_KEY,
    EVENT_LOG_KEY,
    METADATA_KEY,
    WORKER_RESERVATION_KEY,
    LocalSessionStore,
)
from ..storage.base import SessionStore

logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REMOTE_WORKER_STALE_GRACE = timedelta(hours=1)
_SHARED_CLEANUP_STALE_GRACE = timedelta(hours=2)
_SHARED_EVENT_LOG_FLUSH_INTERVAL_SECONDS = 2.0
_SHARED_EVENT_LOG_FLUSH_STATES = frozenset({"completed", "failed", "cancelled"})
_MAINTENANCE_SESSION_ID = "_maintenance"
_CLEANUP_LOCK_KEY = "cleanup.lock"


def validate_session_id(session_id: str) -> str:
    """Validate a session id before it is used in filesystem paths."""
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid session_id")
    if session_id in {".", ".."}:
        raise ValueError("Invalid session_id")
    if "/" in session_id or "\\" in session_id:
        raise ValueError("Invalid session_id")
    return session_id


class SessionManager:
    """Manages pipeline sessions and their artifacts."""

    def __init__(
        self,
        storage_path: Path | str,
        ttl_hours: int = 24,
        store: SessionStore | None = None,
    ):
        """Initialize session manager.

        Args:
            storage_path: Base directory for session storage
            ttl_hours: Time-to-live for sessions in hours
            store: Optional shared storage backend for multi-instance deploys
        """
        self.storage_path = Path(storage_path)
        self.ttl_hours = ttl_hours
        self.store = store or LocalSessionStore(str(self.storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._event_log_flush_at: dict[str, float] = {}
        self._event_log_flush_lock = threading.Lock()
        self._worker_reservation_tokens: dict[str, str] = {}
        self._worker_reservation_tokens_lock = threading.Lock()

    @staticmethod
    def _create_session_layout(session_dir: Path) -> None:
        """Create the directory structure the texture pipeline expects."""
        (session_dir / "input").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "prepared").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "discovery").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "previews").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "generated").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "textures").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "output").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "renders").mkdir(parents=True, exist_ok=True)
        (session_dir / "preview").mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        """Return a validated path guaranteed to remain under storage_path."""
        validate_session_id(session_id)
        storage_root = self.storage_path.resolve()
        session_dir = (storage_root / session_id).resolve()
        if not session_dir.is_relative_to(storage_root):
            raise ValueError("Invalid session_id")
        return session_dir

    def _session_lock_path(self, session_id: str) -> Path:
        """Return a local lock path without creating the session directory."""
        validate_session_id(session_id)
        storage_root = self.storage_path.resolve()
        lock_dir = storage_root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = (lock_dir / f"{session_id}.session.json.lock").resolve()
        if not lock_path.is_relative_to(storage_root):
            raise ValueError("Invalid session_id")
        return lock_path

    def _event_log_lock_path(self, session_id: str) -> Path:
        """Return a local event log lock path without creating the session dir."""
        validate_session_id(session_id)
        storage_root = self.storage_path.resolve()
        lock_dir = storage_root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = (lock_dir / f"{session_id}.event_log.jsonl.lock").resolve()
        if not lock_path.is_relative_to(storage_root):
            raise ValueError("Invalid session_id")
        return lock_path

    def _write_metadata_local(self, session_id: str, metadata: dict[str, Any]) -> None:
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = session_dir / METADATA_KEY
        tmp_path = session_dir / f".{METADATA_KEY}.{uuid.uuid4().hex}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp_path, metadata_path)

    def _ensure_local_session_dir(self, session_id: str) -> Path:
        """Hydrate the local working directory for an existing shared session."""
        session_dir = self._session_dir(session_id)
        metadata_path = session_dir / METADATA_KEY
        if metadata_path.is_file():
            self._create_session_layout(session_dir)
            return session_dir

        if self._uses_shared_store():
            with self._session_lock(session_id):
                if metadata_path.is_file():  # pragma: no cover - hydration race
                    self._create_session_layout(session_dir)
                    return session_dir

                metadata = self.store.get_json(session_id, METADATA_KEY)
                if metadata is None:
                    raise FileNotFoundError(f"Session not found: {session_id}")

                self._create_session_layout(session_dir)
                self._write_metadata_local(session_id, metadata)
                return session_dir

        metadata = self.store.get_json(session_id, METADATA_KEY)
        if metadata is None:
            raise FileNotFoundError(f"Session not found: {session_id}")

        self._create_session_layout(session_dir)  # pragma: no cover
        self._write_metadata_local(session_id, metadata)  # pragma: no cover
        return session_dir  # pragma: no cover

    def _require_session_dir(self, session_id: str) -> Path:
        """Return an existing or hydrated local session dir."""
        session_dir = self._ensure_local_session_dir(session_id)
        metadata_path = session_dir / METADATA_KEY
        if not session_dir.is_dir() or not metadata_path.is_file():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return session_dir

    def _uses_shared_store(self) -> bool:
        store_root = getattr(self.store, "root", None)
        return not (
            self.store.kind == "local"
            and store_root is not None
            and Path(store_root).resolve() == self.storage_path.resolve()
        )

    def uses_shared_store(self) -> bool:
        """Return whether metadata/artifacts are shared across service instances."""
        return self._uses_shared_store()

    def _write_shared_worker_reservation(self, session_id: str) -> str | None:
        if not self._uses_shared_store():
            return None
        metadata = self.get_session_metadata(session_id) or {}
        if metadata.get("status") in {"running", "cancelling"}:
            if not self._remote_worker_metadata_is_stale(metadata, datetime.now(UTC)):
                return None  # pragma: no cover - update/delete race

        owner_token = uuid.uuid4().hex
        marker = {
            "owner_token": owner_token,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "ttl_expires_at": metadata.get("ttl_expires_at"),
        }
        if self.store.put_json_if_absent(session_id, WORKER_RESERVATION_KEY, marker):
            return owner_token

        if self._shared_worker_reservation_active(session_id):
            return None
        if self.store.put_json_if_absent(  # pragma: no cover - stale marker retry
            session_id, WORKER_RESERVATION_KEY, marker
        ):
            return owner_token
        return None  # pragma: no cover - concurrent reservation race

    def _clear_shared_worker_reservation(
        self, session_id: str, owner_token: str | None = None
    ) -> None:
        if not self._uses_shared_store():
            return  # pragma: no cover - delete race
        try:
            if owner_token is not None:
                deleted = self.store.delete_json_if_match(
                    session_id,
                    WORKER_RESERVATION_KEY,
                    lambda marker: marker.get("owner_token") == owner_token,
                )
                if not deleted:
                    logger.info(
                        "Leaving shared worker reservation for %s because owner "
                        "token changed",
                        session_id,
                    )
                    return
                return
            self.store.delete_key(session_id, WORKER_RESERVATION_KEY)
        except Exception as exc:
            logger.warning(
                "Failed to clear shared worker reservation for %s: %s",
                session_id,
                exc,
            )

    def _shared_worker_reservation_active(self, session_id: str) -> bool:
        if not self._uses_shared_store():
            return False
        marker = self.store.get_json(session_id, WORKER_RESERVATION_KEY)
        if marker is None:
            return False

        metadata = self.get_session_metadata(session_id) or {}
        heartbeat = self._latest_timestamp(
            marker.get("updated_at"),
            marker.get("created_at"),
            metadata.get("updated_at"),
        )
        probe = {
            **marker,
            **metadata,
            "updated_at": heartbeat,
            "ttl_expires_at": metadata.get("ttl_expires_at")
            or marker.get("ttl_expires_at"),
        }
        if self._remote_worker_metadata_is_stale(probe, datetime.now(UTC)):
            logger.warning(
                "Clearing stale shared worker reservation for %s",
                session_id,
            )
            owner_token = marker.get("owner_token")
            self._clear_shared_worker_reservation(
                session_id,
                owner_token=owner_token if isinstance(owner_token, str) else None,
            )
            return False
        return True

    @contextmanager
    def _session_lock(self, session_id: str):
        """Acquire an exclusive file lock for a session's metadata.

        Shared stores use this only for same-instance thread/process
        serialization. Cross-instance metadata consistency comes from the
        shared store's conditional-write/CAS operations.

        Raises filelock.Timeout if the lock cannot be acquired within 10s.
        """
        if not self._uses_shared_store():
            self._require_session_dir(session_id)

        lock_path = self._session_lock_path(session_id)
        lock = FileLock(lock_path, timeout=10)
        try:
            with lock:
                yield
        except Timeout:  # pragma: no cover - filelock timing guard
            logger.warning(f"Lock timeout for session {session_id}")
            raise

    @contextmanager
    def worker_lock(self, session_id: str, timeout: float = 10):
        """Acquire a cross-process lock while a worker may write artifacts."""
        lock = self.acquire_worker_lock(session_id, timeout=timeout)
        try:
            yield lock
        finally:
            self.release_worker_lock(lock, session_id)

    def acquire_worker_lock(self, session_id: str, timeout: float = 10) -> FileLock:
        """Acquire the cross-process worker lock and return its handle.

        Routes use this as an accepted-job reservation before returning 202;
        the registry releases it when the queued/running job exits. This keeps
        DELETE/TTL cleanup serialized with jobs even before the executor starts.
        """
        lock_path = self._require_session_dir(session_id) / ".worker.lock"
        lock = FileLock(lock_path, timeout=timeout, thread_local=False)
        try:
            lock.acquire()
            try:
                owner_token = self._write_shared_worker_reservation(session_id)
            except Exception:  # pragma: no cover - reservation write failure
                lock.release()
                raise
            if self._uses_shared_store() and owner_token is None:
                lock.release()
                raise Timeout(str(lock.lock_file))
            if owner_token is not None:
                setattr(lock, "_wu_shared_reservation_token", owner_token)
                with self._worker_reservation_tokens_lock:
                    self._worker_reservation_tokens[session_id] = owner_token
            return lock
        except Timeout:
            logger.warning(f"Worker lock timeout for session {session_id}")
            raise

    def release_worker_lock(self, lock: FileLock, session_id: str) -> None:
        """Release a worker lock handle acquired by acquire_worker_lock."""
        owner_token = getattr(lock, "_wu_shared_reservation_token", None)
        self._clear_shared_worker_reservation(
            session_id,
            owner_token,
        )
        if owner_token is not None:
            with self._worker_reservation_tokens_lock:
                if self._worker_reservation_tokens.get(session_id) == owner_token:
                    self._worker_reservation_tokens.pop(session_id, None)
        try:
            lock.release()
        except Exception:  # pragma: no cover - release failure guard
            logger.exception("Failed to release worker lock for %s", session_id)

    def get_worker_reservation_owner_token(self, session_id: str) -> str | None:
        """Return the local owner token for an active shared worker reservation."""
        with self._worker_reservation_tokens_lock:
            return self._worker_reservation_tokens.get(session_id)

    def _worker_stalled_path(self, session_id: str) -> Path:
        """Return marker path for a worker thread that outlived cancellation."""
        return self._session_dir(session_id) / ".worker.stalled"

    @staticmethod
    def _current_boot_id() -> str | None:
        try:
            return (  # pragma: no cover - platform procfs fallback
                Path("/proc/sys/kernel/random/boot_id")
                .read_text(encoding="utf-8")
                .strip()
            )
        except OSError:  # pragma: no cover - platform procfs fallback
            return None

    @staticmethod
    def _process_start_ticks(pid: int) -> str | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        parts = stat.rsplit(") ", 1)
        if len(parts) != 2:
            return None
        fields = parts[1].split()
        if len(fields) < 20:  # pragma: no cover - malformed procfs stat
            return None
        return fields[19]

    @classmethod
    def _current_stalled_owner(cls) -> dict[str, Any]:
        pid = os.getpid()
        return {
            "pid": pid,
            "boot_id": cls._current_boot_id(),
            "process_start_ticks": cls._process_start_ticks(pid),
        }

    @classmethod
    def _stalled_owner_is_live(cls, marker: dict[str, Any]) -> bool:
        pid = marker.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False

        marker_boot_id = marker.get("boot_id")
        current_boot_id = cls._current_boot_id()
        if (
            isinstance(marker_boot_id, str)
            and current_boot_id is not None
            and marker_boot_id != current_boot_id
        ):
            return False

        marker_start = marker.get("process_start_ticks")
        current_start = cls._process_start_ticks(pid)
        if current_start is None:
            return cls._pid_exists(pid)
        if isinstance(marker_start, str) and marker_start != current_start:
            return False

        return True

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        proc_root = Path("/proc")
        if proc_root.exists() and (proc_root / str(pid)).exists():
            return True
        try:
            os.kill(pid, 0)  # NOSONAR - signal 0 probes liveness only.
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover - process owned by another user
            return True
        return True

    @staticmethod
    def _parse_metadata_datetime(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    @classmethod
    def _latest_timestamp(cls, *values: Any) -> str | None:
        parsed = [cls._parse_metadata_datetime(value) for value in values]
        timestamps = [value for value in parsed if value is not None]
        if not timestamps:
            return None
        return max(timestamps).isoformat()

    @classmethod
    def _shared_marker_is_stale(
        cls,
        marker: dict[str, Any],
        now: datetime,
        grace: timedelta,
    ) -> bool:
        updated_at = cls._parse_metadata_datetime(
            marker.get("updated_at") or marker.get("created_at")
        )
        return updated_at is None or now - updated_at > grace

    @classmethod
    def _remote_worker_metadata_is_stale(
        cls,
        metadata: dict[str, Any],
        now: datetime,
    ) -> bool:
        """Return whether remote running metadata is stale enough to reap.

        A peer instance has no local worker lock to inspect. Treat fresh
        running/cancelling metadata as active, but let TTL cleanup eventually
        reclaim sessions whose owning instance disappeared without writing a
        terminal status.
        """
        updated_at = cls._parse_metadata_datetime(metadata.get("updated_at"))
        if updated_at is not None:
            return now - updated_at > _REMOTE_WORKER_STALE_GRACE

        # Old metadata may not have a useful heartbeat. Fall back to TTL only
        # for that legacy case; fresh sessions without a heartbeat stay active.
        expires_at = cls._parse_metadata_datetime(metadata.get("ttl_expires_at"))
        return expires_at is not None and now > expires_at

    def heartbeat_worker(self, session_id: str, owner_token: str | None = None) -> bool:
        """Refresh the shared worker reservation for a live worker."""
        if not self._uses_shared_store() or owner_token is None:
            return False
        heartbeat_at = datetime.now(UTC).isoformat()

        def updater(marker: dict[str, Any]) -> dict[str, Any] | None:
            if marker.get("owner_token") != owner_token:
                return None  # pragma: no cover - update/delete race
            marker["updated_at"] = heartbeat_at
            return marker

        try:
            updated = self.store.update_json(
                session_id,
                WORKER_RESERVATION_KEY,
                updater,
            )
        except Exception as exc:
            logger.debug(
                "Failed to heartbeat worker reservation for %s: %s", session_id, exc
            )
            return False
        return (
            isinstance(updated, dict)
            and updated.get("owner_token") == owner_token
            and updated.get("updated_at") == heartbeat_at
        )

    def _acquire_shared_cleanup_lock(self) -> str | None:
        if not self._uses_shared_store():
            return None

        owner_token = uuid.uuid4().hex
        marker = {
            "owner_token": owner_token,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if self.store.put_json_if_absent(
            _MAINTENANCE_SESSION_ID,
            _CLEANUP_LOCK_KEY,
            marker,
        ):
            return owner_token

        existing = self.store.get_json(_MAINTENANCE_SESSION_ID, _CLEANUP_LOCK_KEY)
        if isinstance(existing, dict) and self._shared_marker_is_stale(
            existing,
            datetime.now(UTC),
            _SHARED_CLEANUP_STALE_GRACE,
        ):
            existing_owner = existing.get("owner_token")
            if isinstance(existing_owner, str):
                self.store.delete_json_if_match(
                    _MAINTENANCE_SESSION_ID,
                    _CLEANUP_LOCK_KEY,
                    lambda marker: marker.get("owner_token") == existing_owner,
                )
            else:
                self.store.delete_key(_MAINTENANCE_SESSION_ID, _CLEANUP_LOCK_KEY)
            if self.store.put_json_if_absent(
                _MAINTENANCE_SESSION_ID,
                _CLEANUP_LOCK_KEY,
                marker,
            ):
                return owner_token
        return None  # pragma: no cover - cleanup lock owned by another instance

    def _release_shared_cleanup_lock(self, owner_token: str | None) -> None:
        if not self._uses_shared_store() or owner_token is None:
            return  # pragma: no cover - delete race
        try:
            self.store.delete_json_if_match(
                _MAINTENANCE_SESSION_ID,
                _CLEANUP_LOCK_KEY,
                lambda marker: marker.get("owner_token") == owner_token,
            )
        except Exception as exc:
            logger.warning("Failed to release shared cleanup lock: %s", exc)

    def _heartbeat_shared_cleanup_lock(self, owner_token: str | None) -> bool:
        if not self._uses_shared_store() or owner_token is None:
            return True

        heartbeat_at = datetime.now(UTC).isoformat()

        def updater(marker: dict[str, Any]) -> dict[str, Any] | None:
            if marker.get("owner_token") != owner_token:
                return None
            marker["updated_at"] = heartbeat_at
            return marker

        try:
            updated = self.store.update_json(
                _MAINTENANCE_SESSION_ID,
                _CLEANUP_LOCK_KEY,
                updater,
            )
        except Exception as exc:
            logger.warning("Failed to heartbeat shared cleanup lock: %s", exc)
            return False

        return (
            isinstance(updated, dict)
            and updated.get("owner_token") == owner_token
            and updated.get("updated_at") == heartbeat_at
        )

    def mark_worker_stalled(self, session_id: str, reason: str) -> None:
        """Mark that a background worker may still be writing artifacts."""
        marker_path = self._worker_stalled_path(session_id)
        tmp_path = marker_path.with_name(f"{marker_path.name}.{os.getpid()}.tmp")
        marker = {
            "reason": reason,
            "created_at": datetime.now(UTC).isoformat(),
            **self._current_stalled_owner(),
        }
        try:
            tmp_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
            os.replace(tmp_path, marker_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def clear_worker_stalled(self, session_id: str) -> None:
        """Clear the stalled-worker marker after the background thread exits."""
        try:
            self._worker_stalled_path(session_id).unlink(missing_ok=True)
        except ValueError:
            return

    def is_worker_stalled(self, session_id: str) -> bool:
        """Check whether cancellation left a background worker still draining."""
        try:
            marker_path = self._worker_stalled_path(session_id)
        except ValueError:
            return False
        if not marker_path.exists():
            return False

        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Treating unreadable stalled-worker marker as active for %s: %s",
                session_id,
                e,
            )
            return True

        if isinstance(marker, dict) and self._stalled_owner_is_live(marker):
            return True

        marker_path.unlink(missing_ok=True)
        logger.info("Cleared stale stalled-worker marker for %s", session_id)
        return False

    def is_worker_active(self, session_id: str) -> bool:
        """Check whether another worker currently holds the session write lock."""
        if not self.session_exists(session_id):  # pragma: no cover - invalid race
            return False

        if self.is_worker_stalled(session_id):
            return True
        if self._shared_worker_reservation_active(session_id):
            return True

        has_local_metadata = (self._session_dir(session_id) / METADATA_KEY).is_file()
        metadata = (
            self.get_session_metadata(session_id) if self._uses_shared_store() else None
        )
        if metadata and metadata.get("status") in {"running", "cancelling"}:
            if self._remote_worker_metadata_is_stale(metadata, datetime.now(UTC)):
                logger.warning(
                    "Treating stale remote %s session as inactive: %s",
                    metadata.get("status"),
                    session_id,
                )
                if not has_local_metadata:
                    return False
            else:
                logger.debug(
                    "Treating shared %s session as worker-active: %s",
                    metadata.get("status"),
                    session_id,
                )
                return True

        if not has_local_metadata:
            return False

        lock_path = self._session_dir(session_id) / ".worker.lock"
        lock = FileLock(lock_path, timeout=0)
        try:
            with lock:
                return False
        except Timeout:
            return True

    def create_session(
        self, session_id: str, config: dict[str, Any] | None = None
    ) -> Path:
        """Create a new session directory structure.

        Args:
            session_id: Unique session identifier
            config: Optional configuration dict

        Returns:
            Path to session directory
        """
        session_dir = self._session_dir(session_id)

        self.store.init_session(session_id)
        self._create_session_layout(session_dir)

        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": 9,
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

        self._save_metadata(session_id, metadata)
        logger.info(f"Created session: {session_id}")
        return session_dir

    def get_session_dir(self, session_id: str) -> Path:
        """Get path to session directory."""
        return self._session_dir(session_id)

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        try:
            validate_session_id(session_id)
            return self.store.exists(session_id, METADATA_KEY)
        except ValueError:
            return False

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata with retry logic."""
        try:
            validate_session_id(session_id)
        except ValueError:
            return None

        for attempt in range(3):
            try:
                return self.store.get_json(session_id, METADATA_KEY)
            except (OSError, json.JSONDecodeError, BotoCoreError, ClientError) as e:
                if attempt < 2:
                    time.sleep(0.05)
                    continue
                logger.warning(f"Failed to read metadata after 3 attempts: {e}")
                return None

    def _update_metadata(
        self,
        session_id: str,
        updater: Any,
        *,
        update_index: bool = True,
    ) -> dict[str, Any] | None:
        """Update session metadata, using shared-store CAS when available."""
        with self._session_lock(session_id):
            if self._uses_shared_store():
                updated = self.store.update_json(session_id, METADATA_KEY, updater)
                if updated is None:
                    logger.warning(f"Cannot update non-existent session: {session_id}")
                    return None
                self._write_metadata_local(session_id, updated)
                if update_index:
                    self.store.update_session_index(session_id, updated)
                return updated

            metadata = self.get_session_metadata(session_id)
            if not metadata:
                logger.warning(  # pragma: no cover - update/delete race
                    f"Cannot update non-existent session: {session_id}"
                )
                return None  # pragma: no cover - update/delete race
            updated = updater(metadata)
            if updated is None:
                return metadata
            self._save_metadata(session_id, updated, update_index=update_index)
            return updated

    def update_session(
        self,
        session_id: str,
        updates: dict[str, Any],
        *,
        update_index: bool = True,
    ) -> None:
        """Update session metadata."""
        try:

            def _apply_updates(metadata: dict[str, Any]) -> dict[str, Any]:
                metadata.update(updates)
                metadata["updated_at"] = datetime.now(UTC).isoformat()

                created_at = datetime.fromisoformat(metadata["created_at"])
                metadata["elapsed_seconds"] = int(
                    (datetime.now(UTC) - created_at).total_seconds()
                )
                return metadata

            self._update_metadata(
                session_id,
                _apply_updates,
                update_index=update_index,
            )
        except FileNotFoundError:
            logger.warning(f"Cannot update non-existent session: {session_id}")
        except (OSError, BotoCoreError, ClientError, RuntimeError) as exc:
            logger.exception(
                "Failed to update session %s metadata; continuing with last "
                "persisted state: %s",
                session_id,
                exc,
            )
            raise

    def update_step_progress(
        self,
        session_id: str,
        step_name: str,
        progress: dict[str, Any],
    ) -> None:
        """Update progress for current step."""
        self._update_metadata(
            session_id,
            lambda metadata: self._apply_step_progress(metadata, step_name, progress),
        )

    def _apply_step_progress(
        self,
        metadata: dict[str, Any],
        step_name: str,
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        step_info_map = {
            "prepare_uvs": {"display": "Preparing UV Coordinates", "step_num": 1},
            "discover_materials": {
                "display": "Discovering Materials",
                "step_num": 2,
            },
            "plan_textures": {
                "display": "Planning Bounded Texture Work",
                "step_num": 3,
            },
            "generate_prompts": {
                "display": "Generating Texture Prompts",
                "step_num": 4,
            },
            "render_previews": {
                "display": "Rendering Material Previews",
                "step_num": 5,
            },
            "generate_textures": {
                "display": "Generating PBR Textures",
                "step_num": 6,
            },
            "blend_textures": {"display": "Blending Textures", "step_num": 7},
            "apply_textures": {
                "display": "Applying Textures to USD",
                "step_num": 8,
            },
            "render": {"display": "Rendering Final Output", "step_num": 9},
        }

        step_info = step_info_map.get(step_name, {"display": step_name, "step_num": 0})

        current_step_info = metadata.get("current_step")
        if current_step_info and current_step_info.get("name") == step_name:
            started_at = datetime.fromisoformat(current_step_info["started_at"])
            elapsed = int((datetime.now(UTC) - started_at).total_seconds())
            current_step_info["progress"] = progress
            current_step_info["elapsed_seconds"] = elapsed
        else:
            current_step_info = {
                "name": step_name,
                "display_name": step_info["display"],
                "started_at": datetime.now(UTC).isoformat(),
                "progress": progress,
                "elapsed_seconds": 0,
            }

        metadata["current_step"] = current_step_info

        step_num = step_info["step_num"]
        if step_num > 0:
            metadata["overall_progress"]["current_step"] = step_num

        return metadata

    def mark_step_completed(
        self,
        session_id: str,
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Mark a step as completed."""
        self._update_metadata(
            session_id,
            lambda metadata: self._apply_step_completed(metadata, step_name, stats),
        )

    def _apply_step_completed(
        self,
        metadata: dict[str, Any],
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        current_step_info = metadata.get("current_step")
        if current_step_info and current_step_info["name"] == step_name:
            started_at = datetime.fromisoformat(current_step_info["started_at"])
            completed_at = datetime.now(UTC)
            duration = int((completed_at - started_at).total_seconds())

            completed_step = {
                "name": step_name,
                "display_name": current_step_info["display_name"],
                "started_at": current_step_info["started_at"],
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration,
                "stats": stats or {},
            }

            if "completed_steps" not in metadata:
                metadata["completed_steps"] = []
            metadata["completed_steps"].append(completed_step)

            if "timings" not in metadata:
                metadata["timings"] = {}
            metadata["timings"][step_name] = duration

            metadata["current_step"] = None

            completed_count = len(metadata["completed_steps"])
            metadata["overall_progress"]["current_step"] = completed_count

            # Cumulative progress per step completion
            cumulative_percents = {
                "prepare_uvs": 3,
                "discover_materials": 5,
                "plan_textures": 8,
                "generate_prompts": 10,
                "render_previews": 20,
                "generate_textures": 75,
                "blend_textures": 85,
                "apply_textures": 95,
                "render": 100,
            }
            metadata["overall_progress"]["percent"] = cumulative_percents.get(
                step_name, metadata["overall_progress"]["percent"]
            )

            return metadata
        return None

    def add_preview_image(self, session_id: str, image_name: str) -> None:
        """Add a preview image to the session."""

        def _add_preview(metadata: dict[str, Any]) -> dict[str, Any] | None:
            if "preview_images" not in metadata:
                metadata["preview_images"] = []

            if image_name not in metadata["preview_images"]:
                metadata["preview_images"].append(image_name)
                return metadata
            return None

        self._update_metadata(session_id, _add_preview)

    def update_preview_images(self, session_id: str, image_names: list[str]) -> None:
        """Update the list of preview images."""

        def _update_previews(metadata: dict[str, Any]) -> dict[str, Any]:
            metadata["preview_images"] = image_names
            return metadata

        self._update_metadata(session_id, _update_previews)

    def is_cancelled(self, session_id: str) -> bool:
        """Check if session has been cancelled."""
        try:
            validate_session_id(session_id)
        except ValueError:
            return False

        try:
            if self.store.exists(session_id, CANCEL_KEY):
                return True
        except Exception as exc:
            logger.warning(
                "Failed to check shared cancellation marker for %s: %s",
                session_id,
                exc,
            )

        try:
            return (self._session_dir(session_id) / CANCEL_KEY).exists()
        except OSError:  # pragma: no cover - filesystem race
            return False

    def clear_cancellation(self, session_id: str) -> None:
        """Remove the durable `.cancel` marker for a session.

        Callers must hold the cross-process worker lock (see
        ``acquire_worker_lock``) before clearing the marker so a concurrent
        ``request_cancellation`` cannot drop a fresh marker between the
        clear and the new run starting.

        Idempotent: missing marker is a no-op. A missing session directory
        is also tolerated so callers do not have to special-case it after
        delete races; the durable cancellation state simply does not exist.
        """
        try:
            validate_session_id(session_id)
            cancel_file = self._session_dir(session_id) / CANCEL_KEY
        except ValueError:
            return
        try:
            cancel_file.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                f"Failed to clear cancellation marker for {session_id}: {exc}"
            )
        try:
            self.store.delete_key(session_id, CANCEL_KEY)
        except Exception as exc:
            logger.warning(
                f"Failed to clear shared cancellation marker for {session_id}: {exc}"
            )

    def _write_shared_cancel_marker(self, session_id: str) -> None:
        if not self._uses_shared_store():
            return

        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                self.store.put_bytes(session_id, CANCEL_KEY, b"")
                return
            except Exception as exc:
                last_exc = exc
                if attempt == 4:
                    break
                logger.warning(
                    "Failed to write shared cancellation marker for %s "
                    "(attempt %d/5): %s",
                    session_id,
                    attempt + 1,
                    exc,
                )
                time.sleep(0.05 * (attempt + 1))
        raise RuntimeError(
            f"Failed to write shared cancellation marker for {session_id}"
        ) from last_exc

    def request_cancellation(self, session_id: str) -> None:
        """Request cancellation of a running pipeline.

        Idempotent against terminal states: if the session has already
        landed in completed/failed/cancelled (e.g. it finished naturally
        between the cancel route's is_running check and this call), the
        marker is still dropped for any worker still observing it but the
        terminal status is preserved.

        The read-check-write is done atomically under _session_lock so a
        concurrent worker-side update_session(... "completed") cannot
        interleave between the terminal check and the cancelling write.
        """
        if not self.session_exists(session_id):
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return

        self._write_shared_cancel_marker(session_id)
        try:
            cancel_file = self._session_dir(session_id) / CANCEL_KEY
            cancel_file.parent.mkdir(parents=True, exist_ok=True)
            cancel_file.touch()
        except ValueError:  # pragma: no cover - invalid id already checked
            return

        changed = False

        def _mark_cancelling(metadata: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal changed
            current_status = metadata.get("status")
            if current_status in ("completed", "failed", "cancelled"):
                logger.info(
                    f"Cancellation requested but session {session_id} already in "
                    f"terminal state: {current_status}"
                )
                return None

            metadata["status"] = "cancelling"
            metadata["updated_at"] = datetime.now(UTC).isoformat()
            created_at = datetime.fromisoformat(metadata["created_at"])
            metadata["elapsed_seconds"] = int(
                (datetime.now(UTC) - created_at).total_seconds()
            )
            changed = True
            return metadata

        try:
            metadata = self._update_metadata(session_id, _mark_cancelling)
        except FileNotFoundError:  # pragma: no cover - delete race
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return
        if not metadata:
            logger.warning(  # pragma: no cover - delete race
                f"Cannot cancel non-existent session: {session_id}"
            )
            return  # pragma: no cover - delete race

        if changed:
            logger.info(f"Cancellation requested for session: {session_id}")

    def get_artifact_path(self, session_id: str, artifact_type: str) -> Path | None:
        """Get path to a session artifact."""
        session_dir = self.get_session_dir(session_id)

        artifact_map = {
            "materials": session_dir / "cache" / "discovery" / "materials.json",
            "manifest": session_dir / "cache" / "artifacts_manifest.json",
            "output_usd": session_dir / "cache" / "output" / "textured_output.usd",
            "output_usdz": session_dir / "cache" / "output" / "textured_output.usdz",
        }

        path = artifact_map.get(artifact_type)
        if path and path.exists():
            return path

        return None

    def get_artifact_dir(self, session_id: str, artifact_type: str) -> Path | None:
        """Get path to a session artifact directory."""
        session_dir = self.get_session_dir(session_id)

        dir_map = {
            "textures": session_dir / "cache" / "textures",
            "renders": session_dir / "cache" / "renders",
            "generated": session_dir / "cache" / "generated",
            "previews": session_dir / "cache" / "previews",
        }

        path = dir_map.get(artifact_type)
        if path and path.exists():
            return path

        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its artifacts."""
        try:
            session_dir = self.get_session_dir(session_id)
        except ValueError:
            logger.warning(f"Invalid session id for delete: {session_id}")
            return False

        if not self.session_exists(session_id):
            logger.warning(f"Session not found: {session_id}")
            return False

        try:
            with self.worker_lock(session_id, timeout=0):
                if self.is_worker_stalled(session_id):
                    logger.warning(
                        f"Cannot delete stalled worker session: {session_id}"
                    )
                    return False
                with self._session_lock(session_id):
                    self.store.delete_session(session_id)
                    shutil.rmtree(session_dir, ignore_errors=True)
            logger.info(f"Deleted session: {session_id}")
            return True
        except Timeout:
            logger.warning(f"Could not acquire lock to delete session {session_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def cleanup_expired_sessions(self) -> list[str]:
        """Remove sessions past their TTL.

        Returns the list of session IDs whose on-disk directories were
        removed. Caller is responsible for releasing any in-memory state
        (event-bus snapshot/queue) keyed by these IDs -- without that
        follow-up, periodic TTL cleanup would slowly leak stale
        per-session bus state in long-running deployments.
        """
        cleaned: list[str] = []
        now = datetime.now(UTC)
        cleanup_token = self._acquire_shared_cleanup_lock()
        if self._uses_shared_store() and cleanup_token is None:  # pragma: no cover
            logger.debug("Skipping TTL cleanup because another instance owns it")
            return cleaned

        try:
            for metadata in self.list_session_metadata():
                if not self._heartbeat_shared_cleanup_lock(  # pragma: no cover
                    cleanup_token
                ):
                    logger.warning("Stopping TTL cleanup because ownership was lost")
                    break
                session_id = metadata.get("session_id")
                if not isinstance(session_id, str):  # pragma: no cover
                    continue

                try:
                    expires_at = self._parse_metadata_datetime(
                        metadata.get("ttl_expires_at")
                    )
                    if expires_at is None or now <= expires_at:
                        continue

                    # Re-read only expired candidates so cleanup does not fan
                    # out into remote metadata reads for every active session.
                    latest_metadata = self.get_session_metadata(session_id)
                    if not latest_metadata:  # pragma: no cover - delete race
                        continue
                    expires_at = self._parse_metadata_datetime(
                        latest_metadata.get("ttl_expires_at")
                    )
                    if expires_at is None or now <= expires_at:  # pragma: no cover
                        continue

                    if self.is_worker_active(session_id):
                        logger.debug(f"Skipping session {session_id} (worker active)")
                        continue

                    logger.info(f"Cleaning up expired session: {session_id}")
                    if self.delete_session(session_id):
                        cleaned.append(session_id)
                except Timeout:  # pragma: no cover - lock contention race
                    logger.debug(f"Skipping session {session_id} (lock busy)")
                    continue
                except Exception as e:
                    logger.warning(f"Error cleaning session {session_id}: {e}")
                    continue
        finally:
            self._release_shared_cleanup_lock(cleanup_token)

        if cleaned:
            logger.info(f"Cleaned up {len(cleaned)} expired sessions")

        return cleaned

    def _save_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        *,
        update_index: bool = True,
    ) -> None:
        """Save session metadata to disk atomically."""
        self._write_metadata_local(session_id, metadata)
        if self._uses_shared_store():
            self.store.put_json(session_id, METADATA_KEY, metadata)
        if update_index and self._uses_shared_store():
            self.store.update_session_index(session_id, metadata)

    def list_sessions(self) -> list[str]:
        """List all session IDs in the configured store."""
        return self.store.list_sessions()

    def list_session_metadata(self) -> list[dict[str, Any]]:
        """List compact session metadata rows from the configured store."""
        return self.store.list_session_metadata()

    def sync_to_store(self, session_id: str, prefix: str = "") -> int:
        """Sync local session files to shared storage."""
        validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            return 0
        return self.store.sync_from_local(session_id, str(session_dir), prefix=prefix)

    def sync_from_store(self, session_id: str, prefix: str = "") -> int:
        """Hydrate local session files from shared storage."""
        session_dir = self._ensure_local_session_dir(session_id)
        return self.store.sync_to_local(session_id, str(session_dir), prefix=prefix)

    def put_file_to_store(
        self,
        session_id: str,
        key: str,
        file_path: str,
        content_type: str | None = None,
    ) -> None:
        """Copy one file to shared storage."""
        validate_session_id(session_id)
        self.store.put_file(session_id, key, file_path, content_type)

    def store_key_exists(self, session_id: str, key: str) -> bool:
        """Check whether a session key exists in shared storage."""
        try:
            validate_session_id(session_id)
            return self.store.exists(session_id, key)
        except ValueError:
            return False

    def open_store_stream(self, session_id: str, key: str) -> BinaryIO | None:
        """Open a session key from shared storage."""
        validate_session_id(session_id)
        if not self.store.exists(session_id, key):
            return None
        return self.store.open_read(session_id, key)

    def make_store_public_url(
        self,
        session_id: str,
        key: str,
        expires_seconds: int = 3600,
    ) -> str | None:
        """Return a direct public URL for a store key when supported."""
        validate_session_id(session_id)
        if not self.store.exists(session_id, key):
            return None
        url = self.store.make_public_url(
            session_id,
            key,
            expires_seconds=expires_seconds,
        )
        return url

    def list_store_keys(self, session_id: str, prefix: str = "") -> list[str]:
        """List shared storage keys for a session."""
        validate_session_id(session_id)
        return self.store.list_keys(session_id, prefix=prefix)

    def _should_flush_shared_event_log(
        self,
        session_id: str,
        event: dict[str, Any],
    ) -> bool:
        state = event.get("state")
        state_value = getattr(state, "value", state)
        extra = event.get("extra")
        force = state_value in _SHARED_EVENT_LOG_FLUSH_STATES or (
            state_value == "completed"
            and isinstance(extra, dict)
            and bool(extra.get("pipeline_completed"))
        )
        now = time.monotonic()
        with self._event_log_flush_lock:
            last_flush = self._event_log_flush_at.get(session_id)
            if (
                force
                or last_flush is None
                or now - last_flush >= _SHARED_EVENT_LOG_FLUSH_INTERVAL_SECONDS
            ):
                self._event_log_flush_at[session_id] = now
                return True
        return False

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Append an event to the persistent session event log."""
        validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        log_file = session_dir / EVENT_LOG_KEY
        if (session_dir / METADATA_KEY).is_file():
            log_file.parent.mkdir(parents=True, exist_ok=True)
            event_log_lock = FileLock(self._event_log_lock_path(session_id), timeout=10)
            with event_log_lock:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")

                if self._uses_shared_store() and self._should_flush_shared_event_log(
                    session_id,
                    event,
                ):
                    self.store.put_file(
                        session_id,
                        EVENT_LOG_KEY,
                        str(log_file),
                        "application/x-ndjson",
                    )
            return

        if self._uses_shared_store():
            self.store.append_event(session_id, event)

    @staticmethod
    def _read_event_log_file(log_file: Path) -> list[dict[str, Any]]:
        if not log_file.exists():
            return []
        return [
            json.loads(line)
            for line in log_file.read_text(encoding="utf-8").splitlines()
            if line
        ]

    @staticmethod
    def _merge_event_logs(
        *event_logs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[tuple[int, dict[str, Any]]] = []
        seen: set[str] = set()
        order = 0
        for event_log in event_logs:
            for event in event_log:
                key = json.dumps(event, sort_keys=True, separators=(",", ":"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append((order, event))
                order += 1

        def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, float | int, int]:
            order, event = item
            timestamp = event.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=UTC)
                    return (0, parsed.timestamp(), order)
                except ValueError:
                    pass
            return (1, order, order)

        return [event for _order, event in sorted(merged, key=sort_key)]

    def get_event_log(self, session_id: str) -> list[dict[str, Any]]:
        """Read the persisted event log from local disk or shared storage."""
        validate_session_id(session_id)
        log_file = self.get_session_dir(session_id) / EVENT_LOG_KEY
        local_events = self._read_event_log_file(log_file)

        if self._uses_shared_store():
            try:
                shared_events = self.store.get_event_log(session_id)
            except Exception as exc:
                logger.warning(
                    "Failed to read shared event log for %s: %s",
                    session_id,
                    exc,
                )
                return local_events
            return self._merge_event_logs(shared_events, local_events)

        if local_events:
            return local_events
        return self.store.get_event_log(session_id)
