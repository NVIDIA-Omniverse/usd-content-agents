# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import BinaryIO, Protocol

# Standard key for session metadata JSON
METADATA_KEY = "session.json"


class SessionStore(Protocol):
    @property
    def kind(self) -> str: ...  # pragma: no cover - protocol contract only

    # Lifecycle
    async def init_session(self, session_id: str) -> None: ...  # pragma: no cover
    async def delete_session(self, session_id: str) -> None: ...  # pragma: no cover
    async def list_sessions(self, use_cache: bool = True) -> list[str]:
        """List all session IDs in the store.

        Args:
            use_cache: If True, may return cached results for performance.
                       Set to False to force a refresh. (Only affects S3 store)

        Returns:
            List of session IDs
        """
        ...  # pragma: no cover

    def invalidate_sessions_cache(self) -> None:
        """Invalidate the sessions cache.

        For S3 store, clears the cached session list so the next
        list_sessions() call fetches fresh data. For local store, this is a no-op.
        """
        ...  # pragma: no cover

    # Artifacts (images, usd, report, predictions, etc.)
    async def put_bytes(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> None: ...  # pragma: no cover
    async def put_bytes_if_absent(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> bool:
        """Atomically publish bytes only when the key does not exist."""
        ...  # pragma: no cover

    async def put_file(
        self, session_id: str, key: str, file_path: str, content_type: str | None = None
    ) -> None: ...  # pragma: no cover
    async def delete_key(
        self, session_id: str, key: str
    ) -> None: ...  # pragma: no cover
    async def open_read(
        self, session_id: str, key: str
    ) -> BinaryIO: ...  # pragma: no cover
    async def exists(self, session_id: str, key: str) -> bool: ...  # pragma: no cover
    async def list_keys(
        self, session_id: str, prefix: str = ""
    ) -> list[str]: ...  # pragma: no cover

    # Metadata/Status/Events
    async def put_json(
        self, session_id: str, key: str, obj: dict
    ) -> None: ...  # pragma: no cover
    async def get_json(
        self, session_id: str, key: str
    ) -> dict | None: ...  # pragma: no cover
    async def append_event(
        self, session_id: str, event: dict
    ) -> None: ...  # pragma: no cover
    async def get_event_log(
        self, session_id: str
    ) -> list[dict]: ...  # pragma: no cover

    # Public access (for images/files) — may return presigned URL or None if proxy-only
    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None: ...  # pragma: no cover

    # Sync between local and remote storage
    async def sync_to_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Sync files from remote storage to local session directory.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter keys (e.g., "input/")

        Returns:
            Number of files downloaded
        """
        ...  # pragma: no cover

    async def sync_from_local(
        self, session_id: str, local_session_dir: str, prefix: str = ""
    ) -> int:
        """Sync files from local session directory to remote storage.

        Args:
            session_id: Session identifier
            local_session_dir: Path to local session directory
            prefix: Optional prefix to filter files (e.g., "output/")

        Returns:
            Number of files synced
        """
        ...  # pragma: no cover

    # Local cache cleanup (for remote stores like S3)
    async def cleanup_stale_local_sessions(
        self,
        local_storage_path: str,
        max_age_hours: float = 24.0,
        skip_session_ids: set[str] | None = None,
    ) -> int:
        """Clean up stale local session directories.

        For remote stores (S3), syncs sessions to remote and removes local cache
        if the session hasn't been updated for longer than max_age_hours.

        For local stores, this is a no-op since files are already in their
        final location.

        Args:
            local_storage_path: Root path where local sessions are stored
            max_age_hours: Maximum age in hours before a session is considered stale
            skip_session_ids: Session IDs that must not be removed locally

        Returns:
            Number of sessions cleaned up
        """
        ...  # pragma: no cover
