# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
from pathlib import Path
from typing import IO

from world_understanding.utils.artifacts import (
    ArtifactPathError,
    append_bytes_to_confined,
    confined_artifact_exists,
    copy_open_file_to_confined,
    delete_confined_file,
    is_pipeline_temp_path,
    iter_open_regular_files,
    list_confined_artifact_keys,
    open_confined_directory,
    open_confined_directory_at,
    open_confined_lock_file,
    open_confined_regular_file,
    open_held_confined_artifact,
    open_regular_file_no_follow,
    prune_confined_snapshot,
    read_confined_artifact_bytes,
    remove_confined_tree,
    validated_artifact_relative_key,
    write_bytes_to_confined,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.session_paths import (
    confined_session_path,
    confined_storage_child_path,
    is_safe_session_id,
)

from .base import SessionStore

logger = logging.getLogger(__name__)


class LocalSessionStore(SessionStore):
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)

    @property
    def kind(self) -> str:
        return "local"

    def _session_dir(self, session_id: str) -> Path:
        if is_safe_session_id(session_id):
            return confined_session_path(self.root, session_id)
        return confined_storage_child_path(self.root, session_id)

    def _read_session_dir(self, session_id: str) -> Path | None:
        """Resolve a read root, treating an escaped UUID alias as absent."""
        try:
            return self._session_dir(session_id)
        except ValueError:
            return None

    async def init_session(self, session_id: str) -> None:
        self._session_dir(session_id)
        with open_confined_directory(self.root, create=True) as root_descriptor:
            with open_confined_directory_at(
                root_descriptor,
                session_id,
                create=True,
            ):
                pass

    async def delete_session(self, session_id: str) -> None:
        for attempt in range(3):
            try:
                remove_confined_tree(self._session_dir(session_id), self.root)
                return
            except OSError:
                if attempt == 2:
                    raise
                log_durable_failure(
                    logger,
                    "session_local_delete_retry_failed",
                    phase=FailurePhase.ROLLBACK,
                    retryable=True,
                )
                await asyncio.sleep(0.5 * (attempt + 1))

    async def list_sessions(self, use_cache: bool = True) -> list[str]:
        """List all session IDs in the local store.

        Args:
            use_cache: Ignored for local storage (filesystem is fast,
                       no caching needed)

        Returns:
            List of session IDs (directory names that contain session.json)
        """
        sessions: list[str] = []
        if not self.root.exists():
            return sessions

        for listed_path in self.root.iterdir():
            try:
                session_dir = self._session_dir(listed_path.name)
            except ValueError:
                continue
            if confined_artifact_exists(session_dir, "session.json"):
                sessions.append(listed_path.name)

        return sessions

    def invalidate_sessions_cache(self) -> None:
        """No-op for local storage - no caching needed."""
        pass

    async def put_bytes(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> None:
        del content_type
        with open_confined_directory(
            self._session_dir(session_id),
            create=True,
        ) as destination_descriptor:
            write_bytes_to_confined(
                destination_descriptor,
                key,
                data,
                file_mode=0o600,
            )

    async def put_bytes_if_absent(
        self, session_id: str, key: str, data: bytes, content_type: str | None = None
    ) -> bool:
        del content_type
        with open_confined_directory(
            self._session_dir(session_id),
            create=True,
        ) as destination_descriptor:
            return write_bytes_to_confined(
                destination_descriptor,
                key,
                data,
                overwrite=False,
                file_mode=0o600,
            )

    async def compare_and_swap_bytes(
        self,
        session_id: str,
        key: str,
        expected: bytes,
        replacement: bytes | None,
        content_type: str | None = None,
    ) -> bool:
        """Compare and swap under a stable cross-process file lock."""

        del content_type
        canonical_key = validated_artifact_relative_key(key)
        key_parts = canonical_key.split("/")
        lock_key = "/".join((*key_parts[:-1], f".{key_parts[-1]}.cas.lock"))
        with open_confined_directory(
            self._session_dir(session_id),
            create=True,
        ) as session_descriptor:
            with open_confined_lock_file(
                session_descriptor,
                lock_key,
            ) as lock_descriptor:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                try:
                    try:
                        with open_confined_regular_file(
                            session_descriptor,
                            canonical_key,
                        ) as (stream, _metadata):
                            current = stream.read()
                    except FileNotFoundError:
                        return False
                    if current != expected:
                        return False
                    if replacement is None:
                        return delete_confined_file(
                            session_descriptor,
                            canonical_key,
                            missing_ok=False,
                        )
                    write_bytes_to_confined(
                        session_descriptor,
                        canonical_key,
                        replacement,
                        file_mode=0o600,
                    )
                    return True
                finally:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)

    async def put_file(
        self, session_id: str, key: str, file_path: str, content_type: str | None = None
    ) -> None:
        del content_type
        with open_regular_file_no_follow(file_path) as (source, metadata):
            with open_confined_directory(
                self._session_dir(session_id),
                create=True,
            ) as destination_descriptor:
                copy_open_file_to_confined(
                    destination_descriptor,
                    key,
                    source,
                    metadata,
                    overwrite=True,
                )

    async def open_read(self, session_id: str, key: str) -> IO[bytes]:
        base = self._read_session_dir(session_id)
        if base is None:
            raise FileNotFoundError(key)
        try:
            return open_held_confined_artifact(base, key).stream
        except (
            ArtifactPathError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            raise FileNotFoundError(key) from None

    async def exists(self, session_id: str, key: str) -> bool:
        base = self._read_session_dir(session_id)
        return base is not None and confined_artifact_exists(base, key)

    async def delete_key(self, session_id: str, key: str) -> None:
        try:
            with open_confined_directory(
                self._session_dir(session_id),
            ) as session_descriptor:
                delete_confined_file(session_descriptor, key)
        except FileNotFoundError:
            pass

    async def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        base = self._read_session_dir(session_id)
        if base is None or is_pipeline_temp_path(prefix):
            return []
        return [
            key
            for key in list_confined_artifact_keys(base, prefix=prefix)
            if not key.endswith(".cas.lock")
        ]

    async def put_json(self, session_id: str, key: str, obj: dict) -> None:
        await self.put_bytes(
            session_id, key, json.dumps(obj).encode("utf-8"), "application/json"
        )

    async def get_json(self, session_id: str, key: str) -> dict | None:
        base = self._read_session_dir(session_id)
        if base is None:
            return None
        try:
            data = read_confined_artifact_bytes(base, key)
            return json.loads(data)
        except (
            ArtifactPathError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            return None

    async def append_event(self, session_id: str, event: dict) -> None:
        line = (json.dumps(event) + "\n").encode("utf-8")
        with open_confined_directory(
            self._session_dir(session_id),
            create=True,
        ) as session_descriptor:
            append_bytes_to_confined(session_descriptor, "events.jsonl", line)

    async def get_event_log(self, session_id: str) -> list[dict]:
        base = self._read_session_dir(session_id)
        if base is None:
            return []
        try:
            text = read_confined_artifact_bytes(base, "events.jsonl").decode("utf-8")
            return [json.loads(line) for line in text.splitlines()]
        except (
            ArtifactPathError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ):
            return []

    async def make_public_url(
        self, session_id: str, key: str, expires_seconds: int = 3600
    ) -> str | None:
        return None

    async def sync_to_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        """Copy files from store to local dir (no-op if they are the same path).

        With ``overwrite``, the requested prefix becomes a source snapshot:
        copied files are atomically replaced before stale local entries are pruned.
        The return value counts copied files only, not pruned files.
        """
        store_dir = self._read_session_dir(session_id)
        if store_dir is None:
            return 0
        return self._copy_local_snapshot(
            store_dir,
            Path(local_session_dir),
            prefix,
            overwrite=overwrite,
        )

    async def sync_from_local(
        self,
        session_id: str,
        local_session_dir: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        """Copy files from local dir to store (no-op if they are the same path)."""
        return self._copy_local_snapshot(
            Path(local_session_dir),
            self._session_dir(session_id),
            prefix,
            overwrite=overwrite,
        )

    @staticmethod
    def _copy_local_snapshot(
        source_dir: Path,
        destination_dir: Path,
        prefix: str,
        *,
        overwrite: bool,
    ) -> int:
        """Copy held regular files between descriptor-confined local roots."""

        try:
            with open_confined_directory(source_dir) as source_descriptor:
                with open_confined_directory(
                    destination_dir,
                    create=True,
                ) as destination_descriptor:
                    source_identity = os.fstat(source_descriptor)
                    destination_identity = os.fstat(destination_descriptor)
                    if (
                        source_identity.st_dev == destination_identity.st_dev
                        and source_identity.st_ino == destination_identity.st_ino
                    ):
                        return 0
                    count = 0
                    source_relative_keys: set[str] = set()
                    for artifact in iter_open_regular_files(
                        source_descriptor,
                        prefix=prefix,
                    ):
                        source_relative_keys.add(artifact.relative_key)
                        if copy_open_file_to_confined(
                            destination_descriptor,
                            artifact.relative_key,
                            artifact.stream,
                            artifact.metadata,
                            overwrite=overwrite,
                        ):
                            count += 1
                    if overwrite:
                        prune_confined_snapshot(
                            destination_descriptor,
                            prefix,
                            source_relative_keys,
                        )
                    return count
        except FileNotFoundError:
            return 0

    async def cleanup_stale_local_sessions(
        self, local_storage_path: str, max_age_hours: float = 24.0
    ) -> int:
        """No-op for local storage - no remote sync needed.

        Args:
            local_storage_path: Root path where local sessions are stored
            max_age_hours: Maximum age in hours (ignored)

        Returns:
            0 (no cleanup needed for local storage)
        """
        return 0
